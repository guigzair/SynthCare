from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
import json
from peft import AutoPeftModelForCausalLM


model_path = "mistral_tuning_adapter"
base_model_name = "mistralai/Ministral-3-3B-Instruct-2512"
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",          # Automatically chooses GPU/CPU
    torch_dtype=torch.bfloat16, # Matches Mistral's native precision
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
device = "cuda" if torch.cuda.is_available() else "cpu"

class ChatInterface:
    """Chat-like interface for the model"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def chat(self, user_message):
        """Get response from model"""
        inputs = self.tokenizer.encode(user_message, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

model.eval()

chat = ChatInterface(model, tokenizer, device)
# Interactive chat
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = chat.chat(user_input)
    print(f"Assistant: {response}\n")
