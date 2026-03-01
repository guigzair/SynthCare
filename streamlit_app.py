import streamlit as st
import torch
from Mistral_GAN_example import MistralGAN
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
import json
from peft import AutoPeftModelForCausalLM

st.set_page_config(page_title="Mistral GAN", page_icon="🤖")

st.title("🤖 SynthCare: GDPR-Compliant Synthetic Health Data Generator")
st.markdown("Generate realistic instruction-context-output data using adversarial training")

st.divider()

# Load model
@st.cache_resource
def load_models():
    return MistralGAN(model_name="mistralai/Ministral-3-3B-Instruct-2512")

st.header("1️⃣ Load Model")
if st.button("Load Mistral GAN Models"):
    with st.spinner("Loading models... This takes ~1-2 minutes"):
        model_path = "./logs/mistral_tuning_adapter"
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

        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.device = device
        st.success("✅ Models loaded!")

st.divider()

st.header("2️⃣ Generate Samples")

if 'model' not in st.session_state:
    st.warning("⚠️ Please load the models first!")
else:
    col1, col2 = st.columns(2)
    with open('prompt.txt', 'r') as file:
        prompt = file.read()
    prompts = [prompt] * 1

    # with col2:
    #     max_length = st.slider("Max length", 50, 500, 150)
    #     st.write("")  # spacing
    
    if st.button("🎲 Generate sample", use_container_width=True):
        with st.spinner("Generating sample..."):
            input_ids = st.session_state.tokenizer(prompts, return_tensors="pt", padding=True).to(st.session_state.device)
            output = st.session_state.model.generate(
                **input_ids,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id= st.session_state.tokenizer.eos_token_id,
            )
        
        st.success(f"✅ Generated sample!")
        
        with st.expander(f"📄 Sample"):
                prompt_length = len(prompts[0]) - 10  # Adjust for special tokens if needed
                print(f"Prompt length: {prompt_length}")
                output = st.session_state.tokenizer.decode(output[0], skip_special_tokens=True)
                output = output[prompt_length:]  # Remove the prompt from the output
                st.text_area(
                    "Text",
                    value = output,
                    height=500,
                    disabled=True,
                    key=f"sample"
                )

        # # Display samples
        # for i, sample in enumerate(samples, 1):
        #     with st.expander(f"📄 Sample {i}"):
        #         st.text_area(
        #             "Text",
        #             value=sample['text'],
        #             height=150,
        #             disabled=True,
        #             key=f"sample_{i}"
        #         )

st.divider()

st.header("ℹ️ About")
st.markdown("""
### Architecture
- **Generator**: Ministral-3-3B (Causal LM)
- **Discriminator**: Ministral-3-3B (Sequence Classification)
- **Optimization**: LoRA fine-tuning + 4-bit quantization

### How It Works
1. Generator creates synthetic instruction-context-output triplets
2. Discriminator learns to distinguish real vs synthetic data
3. Both improve together through adversarial training

### Resources
- [Mistral AI](https://mistral.ai)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Streamlit Docs](https://docs.streamlit.io)
""")