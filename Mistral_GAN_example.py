import json
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import Adam
import numpy as np


class MistralGAN:
    def __init__(self, model_name = "mistralai/Ministral-3-3B-Instruct-2512"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prompt load text
        with open('prompt.txt', 'r') as file:
            self.prompt = file.read()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generator model
        print("Loading generator model...")
        self.generator = Mistral3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=FineGrainedFP8Config(dequantize=True)
            )
        
        # Apply LoRA to generator
        gen_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        self.generator = get_peft_model(self.generator, gen_peft_config)
        
        # Discriminator model (sequence classification)
        print("Loading discriminator model...")
        self.discriminator = Mistral3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=FineGrainedFP8Config(dequantize=True)
            )
        
        
        # Apply LoRA to discriminator
        disc_peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        self.discriminator = get_peft_model(self.discriminator, disc_peft_config)
        
        # Tokenizer and device setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move models to device
        self.generator.train()
        self.discriminator.train()
    
    def generate_synthetic_data(self, num_samples=5, max_length=512):
        """
        Generate synthetic instruction-context-output triplets
        """
        self.generator.eval()
        synthetic_data = []
        
        with torch.no_grad():
            prompts = [self.prompt] * num_samples
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
                
            output = self.generator.generate(
                **input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_text = self.tokenizer.batch_decode(
                output,
                skip_special_tokens=True
            )
            for i in range(num_samples):
                synthetic_data.append({
                    'text': generated_text[i],
                    'label': 0  # 0 = synthetic/fake
                })
        
        self.generator.train()
        return synthetic_data
    
    def discriminator_forward(self, texts, labels):
        """
        Get discriminator predictions and loss
        """
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        labels_tensor = torch.tensor(labels).to(self.device)
        
        outputs = self.discriminator(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels_tensor
        )
        
        return outputs.loss, outputs.logits
    
    def train_step(self, real_texts, real_labels, synthetic_texts, synthetic_labels, 
                   gen_optimizer, disc_optimizer):
        """
        Single GAN training step with proper tokenization
        """
        # --- Train Discriminator ---
        disc_optimizer.zero_grad()
        
        # Tokenize real data
        real_inputs = self.tokenizer(
            real_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        
        real_labels_tensor = torch.tensor(real_labels).to(self.device)
        real_labels_tensor = real_labels_tensor.unsqueeze(1).expand(-1, real_inputs['input_ids'].shape[1])  # Expand to match sequence length
        real_loss = self.discriminator(
            input_ids=real_inputs['input_ids'],
            attention_mask=real_inputs['attention_mask'],
            labels=real_labels_tensor
        ).loss
        print("Real loss computed: ", real_loss)
        
        # Tokenize synthetic data
        synthetic_inputs = self.tokenizer(
            synthetic_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        synthetic_labels_tensor = torch.tensor(synthetic_labels).to(self.device)
        synthetic_labels_tensor = synthetic_labels_tensor.unsqueeze(1).expand(-1, synthetic_inputs['input_ids'].shape[1])  # Expand to match sequence length
        synthetic_loss = self.discriminator(
            input_ids=synthetic_inputs['input_ids'],
            attention_mask=synthetic_inputs['attention_mask'],
            labels=synthetic_labels_tensor
        ).loss
        print("Synthetic loss computed: ", synthetic_loss)
        
        disc_loss = (real_loss + synthetic_loss) / 2
        disc_loss.backward()
        disc_optimizer.step()
        
        # --- Train Generator ---
        gen_optimizer.zero_grad()
        
        # Generator input: start with real data and let it learn to extend
        # Tokenize real texts for generator training
        gen_inputs = self.tokenizer(
            synthetic_texts,  # Use synthetic text as context
            truncation=True,
            max_length=480,  # Leave room for generation
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass through generator
        outputs = self.generator(
            input_ids=gen_inputs['input_ids'],
            attention_mask=gen_inputs['attention_mask'],
            labels=gen_inputs['input_ids']  # Predict next tokens
        )
        gen_loss, gen_logits = outputs.loss, outputs.logits
        print("Generator LM loss computed: ", gen_loss)
        
        # Add discriminator feedback: make discriminator think synthetic data is real
        disc_for_gen_inputs = self.tokenizer(
            synthetic_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        fool_labels = torch.ones(len(synthetic_texts), dtype=torch.long).to(self.device)
        fool_labels = fool_labels.unsqueeze(1).expand(-1, disc_for_gen_inputs['input_ids'].shape[1])  # Expand to match sequence length
        disc_loss_for_gen = self.discriminator(
            input_ids=disc_for_gen_inputs['input_ids'],
            attention_mask=disc_for_gen_inputs['attention_mask'],
            labels=fool_labels
        ).loss
        print("Discriminator loss for generator computed: ", disc_loss_for_gen)
        
        # Combined loss: generation quality + fooling discriminator
        combined_gen_loss = gen_loss + 0.5 * disc_loss_for_gen
        combined_gen_loss.backward()
        gen_optimizer.step()
        
        return {
            'disc_loss': disc_loss.item(),
            'gen_loss': combined_gen_loss.item(),
            'gen_lm_loss': gen_loss.item(),
            'gen_fool_loss': disc_loss_for_gen.item(),
        }
    
    def tokenize_function(self, examples):
        outputs =  self.tokenizer(examples['text'], 
                                padding="max_length",   # Pads short sequences with 0s
                                truncation=True,        # Cuts off sequences longer than max_length
                                max_length=512,         # Common starting point for Mistral
                                return_tensors=None     # Crucial: Keep as list for the .map() function
                            )
        # THE FIX: Copy input_ids to labels
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    def load_formatted_dataset(self, real_data_path = "clinical_stories.jsonl"):
        """
        Load and tokenize real dataset
        """
        dataset = load_dataset("json", data_files={"train": real_data_path}, split="train")

        def format_instruction_prompt(example):
            # Mistral uses <s> [INST] Instruction + Context [/INST] Response </s>
            text = (
                f"<s>[INST] {example['instruction']}\n\n"
                f"## CONTEXT\n{example['context']} [/INST] "
                f"{example['output']} </s>"
            )
            return {"text": text}
        # formatted_data = map(format_instruction_prompt, dataset)
        formatted_data = [format_instruction_prompt(item) for item in dataset]
        formatted_data = Dataset.from_dict({'text': [item['text'] for item in formatted_data]})

        # tokenized_dataset = dataset.map(self.tokenize_function)
        return formatted_data

    def train(self, real_data_path = "clinical_stories.jsonl", num_epochs=3, learning_rate=2e-4):
        print("Starting GAN training...")
        """
        Train the GAN
        """
        dataset = self.load_formatted_dataset(real_data_path=real_data_path)
        
        # Setup optimizers
        gen_optimizer = Adam(self.generator.parameters(), lr=learning_rate)
        disc_optimizer = Adam(self.discriminator.parameters(), lr=learning_rate)
        
        print(f"\nStarting GAN training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_losses = {'gen': [], 'disc': []}
            
            # Train on batches of real data
            batch_size = 8
            for i in range(0, len(dataset), batch_size):
                batch_real_texts = dataset[i:i+batch_size]['text']
                batch_real_labels = [1] * len(batch_real_texts)  # 1 = real
                
                # Generate synthetic batch
                synthetic_samples = self.generate_synthetic_data(
                    num_samples=len(batch_real_texts)
                )
                batch_synthetic_texts = [s['text'] for s in synthetic_samples]
                batch_synthetic_labels = [0] * len(batch_synthetic_texts) # 0 = synthetic/fake
                print("Batch sizes - Real: {}, Synthetic: {}".format(len(batch_real_texts), len(batch_synthetic_texts)))
                
                print(f"Epoch {epoch+1}, Batch {i//batch_size+1}: ")
                # Train step
                losses = self.train_step(
                    batch_real_texts,
                    batch_real_labels,
                    batch_synthetic_texts,
                    batch_synthetic_labels,
                    gen_optimizer,
                    disc_optimizer
                )
                
                epoch_losses['gen'].append(losses['gen_loss'])
                epoch_losses['disc'].append(losses['disc_loss'])
            
            avg_gen_loss = np.mean(epoch_losses['gen'])
            avg_disc_loss = np.mean(epoch_losses['disc'])
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Generator Loss: {avg_gen_loss:.4f}, "
                  f"Discriminator Loss: {avg_disc_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                self.generator.save_pretrained(f"./mistral_gan_generator_epoch_{epoch+1}")
                self.discriminator.save_pretrained(f"./mistral_gan_discriminator_epoch_{epoch+1}")
        
        print("\nTraining complete!")
    
    def generate_dataset(self, num_samples=100, output_path="synthetic_data.jsonl"):
        """
        Generate a full synthetic dataset
        """
        print(f"Generating {num_samples} synthetic samples...")

        synthetic_data = self.generate_synthetic_data(num_samples=num_samples)       

        # Save to JSONL
        with open(output_path, 'w') as f:
            for item in synthetic_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(synthetic_data)} samples to {output_path}")
        return synthetic_data


# Usage example
if __name__ == "__main__":
    # Initialize GAN
    gan = MistralGAN(model_name="mistralai/Ministral-3-3B-Instruct-2512")
    
    # Train the GAN
    gan.train(
        real_data_path="clinical_stories.jsonl",
        num_epochs=3,
        learning_rate=2e-3
    )
    
    # Generate synthetic dataset
    synthetic_data = gan.generate_synthetic_data(
        num_samples=20
    )