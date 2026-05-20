import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist")
st.write("تحليل ذكي بدون تكرار")

# -----------------------
# BLIP
# -----------------------
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, blip_model = load_caption_model()

# -----------------------
# FLAN-T5 (بدون pipeline)
# -----------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, llm_model = load_llm()

# -----------------------
# تحليل AI
# -----------------------
def ai_fashion_analysis(description):

    prompt = f"""
أنت خبير أزياء.
حلل الإطلالة التالية بشكل مختلف حسب المحتوى.

الوصف:
{description}

اكتب:
- تقييم من 10
- نوع الستايل
- نقاط القوة
- نقاط التحسين
- الانطباع العام

باللغة العربية.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=200)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return result

# -----------------------
# رفع الصورة
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("تحليل"):

        with st.spinner("جاري التحليل..."):

            # وصف الصورة
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = blip_model.generate(**inputs, max_new_tokens=50)

            description = processor.decode(out[0], skip_special_tokens=True)

            # تحليل ذكي
            analysis = ai_fashion_analysis(description)

            st.subheader("📌 الوصف:")
            st.write(description)

            st.subheader("🧠 التحليل:")
            st.write(analysis)