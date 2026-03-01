# Install required packages
# !pip install transformers datasets peft torch bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
import json

##########################################################################
####################### 1. Load a Model
##########################################################################

model_name = "mistralai/Ministral-3-3B-Instruct-2512"
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=FineGrainedFP8Config(dequantize=True)
)

##########################################################################
####################### 2. Configure LoRA (Low-Rank Adaptation)
##########################################################################
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model_generator = get_peft_model(model, peft_config)
model_discriminator = get_peft_model(model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"

##########################################################################
######################## 3. Prepare your dataset jsonl format
##########################################################################

dataset = load_dataset("json", data_files={"train": "clinical_stories.jsonl"}, split="train")
def format_prompt(example):
    return {
        'text': f"Instruction: {example['instruction']}\nContext: {example['context']}\nOutput: {example['output']}"
    }

def format_instruction_prompt(example):
    # Mistral uses <s> [INST] Instruction + Context [/INST] Response </s>
    text = (
        f"<s>[INST] {example['instruction']}\n\n"
        f"## CONTEXT\n{example['context']} [/INST] "
        f"{example['output']} </s>"
    )
    return {"text": text}

formatted_data = [format_instruction_prompt(item) for item in dataset]
dataset = Dataset.from_dict({'text': [item['text'] for item in formatted_data]})

##########################################################################
####################### 4. Tokenize
##########################################################################
def tokenize_function(examples):
    outputs =  tokenizer(examples['text'], 
                        padding="max_length",   # Pads short sequences with 0s
                        truncation=True,        # Cuts off sequences longer than max_length
                        max_length=512,         # Common starting point for Mistral
                        return_tensors=None     # Crucial: Keep as list for the .map() function
                    )
    # THE FIX: Copy input_ids to labels
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize_function)

##########################################################################
######################## 5. Training arguments
##########################################################################

training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
)

##########################################################################
######################### 6. Train
##########################################################################

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    max_length=512  # Optional: pad to fixed length instead of max in batch
)

trainer = Trainer(
    model=model_generator,  # Use the generator model for training
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

##########################################################################
######################### 7. Save the adapter
##########################################################################
model_generator.save_pretrained("./mistral_tuning_adapter")