import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# =====================================================================
# SECTION 1: SETTING UP THE WEB INTERFACE
# =====================================================================
st.set_page_config(page_title="AI Image Detector", page_icon="🖼️")
st.title("🔍 AI Image Detector")
st.write("Upload an image, and the AI model will predict if it's Real or AI-generated!")

# =====================================================================
# SECTION 2: LOADING THE AI MODEL (CACHED)
# =====================================================================
@st.cache_resource
def load_detector_model():
    model_name = "Smogy/SMOGY-Ai-images-detector"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_detector_model()

# =====================================================================
# SECTION 3: UPLOADING & PROCESSING THE IMAGE
# =====================================================================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image to the user
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("---")
    st.write("🔄 Analyzing image patterns... please wait.")
    
    # Preprocess the image for the model
    inputs = processor(images=image, return_tensors="pt")

# =====================================================================
# SECTION 4: MAKING THE PREDICTION & DISPLAYING RESULTS
# =====================================================================
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert raw outputs (logits) into percentage probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Dynamically extract and match labels to their exact scores
    ai_prob = 0.0
    real_prob = 0.0

    for idx, prob in enumerate(probabilities):
        label_text = model.config.id2label[idx].lower()
        percentage = prob.item() * 100
        
        # Check for any variation of AI or Real label keywords
        if any(kw in label_text for kw in ["ai", "fake", "synthetic", "artificial"]):
            ai_prob = percentage
        elif any(kw in label_text for kw in ["real", "human"]):
            real_prob = percentage

    # Create two layout columns on the screen
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Probability it is REAL", value=f"{real_prob:.2f}%")
        
    with col2:
        st.metric(label="Probability it is AI-Generated", value=f"{ai_prob:.2f}%")
        
    # Final Verdict Banner
    if ai_prob > real_prob:
        st.error(f"🚨 Verdict: This image looks AI-generated ({ai_prob:.1f}% confidence).")
    else:
        st.success(f"✅ Verdict: This image looks Human-made/Real ({real_prob:.1f}% confidence).")