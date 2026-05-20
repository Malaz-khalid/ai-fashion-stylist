import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist")
st.write("تحليل ذكي كامل بدون قواعد ثابتة")

# -----------------------
# BLIP (وصف الصورة)
# -----------------------
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_caption_model()

# -----------------------
# AI لتوليد التحليل (بدون قواعد)
# -----------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="gpt2"
    )

llm = load_llm()

# -----------------------
# رفع الصورة
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

# -----------------------
# تحليل AI (بدون rules)
# -----------------------
def ai_fashion_analysis(description):

    prompt = f"""
أنت خبير موضة.
حلل هذا الوصف لإطلالة شخص واعطه:

- تقييم من 10
- نوع الستايل
- نقاط القوة
- نقاط التحسين
- الانطباع العام

الوصف:
{description}

الإجابة بالعربية وبشكل منسق:
"""

    result = llm(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]

    return result

# -----------------------
# التشغيل
# -----------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المدخلة", use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل بالذكاء الاصطناعي..."):

            # 1. وصف الصورة
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            description = processor.decode(out[0], skip_special_tokens=True)

            # 2. تحليل AI كامل بدون قواعد
            analysis = ai_fashion_analysis(description)

            # -----------------------
            # النتائج
            # -----------------------
            st.subheader("📌 الوصف:")
            st.write(description)

            st.subheader("🧠 التحليل الذكي:")
            st.write(analysis)