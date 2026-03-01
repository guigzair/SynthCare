import streamlit as st
import torch
from Mistral_GAN_example import MistralGAN

st.set_page_config(page_title="Mistral GAN", page_icon="🤖")

st.title("🤖 Mistral GAN - Synthetic Data Generator")
st.markdown("Generate realistic instruction-context-output data using adversarial training")

st.divider()

# Load model
@st.cache_resource
def load_models():
    return MistralGAN(model_name="mistralai/Ministral-3-3B-Instruct-2512")

st.header("1️⃣ Load Model")
if st.button("Load Mistral GAN Models"):
    with st.spinner("Loading models... This takes ~1-2 minutes"):
        gan = load_models()
        st.session_state.gan = gan
        st.success("✅ Models loaded!")

st.divider()

st.header("2️⃣ Generate Samples")

if 'gan' not in st.session_state:
    st.warning("⚠️ Please load the models first!")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        prompt = st.text_input("Prompt:", value="Instruction:")
        num_samples = st.slider("Number of samples", 1, 10, 3)
    
    with col2:
        max_length = st.slider("Max length", 50, 500, 150)
        st.write("")  # spacing
    
    if st.button("🎲 Generate", use_container_width=True):
        with st.spinner("Generating samples..."):
            samples = st.session_state.gan.generate_synthetic_data(
                prompt=prompt,
                num_samples=num_samples,
                max_length=max_length
            )
        
        st.success(f"✅ Generated {len(samples)} samples!")
        
        # Display samples
        for i, sample in enumerate(samples, 1):
            with st.expander(f"📄 Sample {i}"):
                st.text_area(
                    "Text",
                    value=sample['text'],
                    height=150,
                    disabled=True,
                    key=f"sample_{i}"
                )

st.divider()

st.header("ℹ️ About")
st.markdown("""
### Architecture
- **Generator**: Mistral-7B (Causal LM)
- **Discriminator**: Mistral-7B (Sequence Classification)
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