import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist")
st.write("تحليل ذكي حقيقي لكل صورة")

# -----------------------
# BLIP لوصف الصورة
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
# AI تحليل ذكي (أفضل من GPT-2)
# -----------------------
@st.cache_resource
def load_ai():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

llm = load_ai()

# -----------------------
# رفع الصورة
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

# -----------------------
# تحليل ذكي بدون تكرار
# -----------------------
def ai_fashion_analysis(description):

    prompt = f"""
أنت خبير ستايل أزياء محترف جداً.

حلل الإطلالة التالية بشكل مختلف تماماً حسب المحتوى.

الوصف:
{description}

اكتب تحليل فريد لكل حالة ويجب أن يتغير حسب الصورة.

اكتب:
1- تقييم من 10
2- نوع الستايل
3- نقاط القوة (مخصصة للصورة فقط)
4- نقاط التحسين (مخصصة للصورة فقط)
5- الانطباع العام

اكتب بالعربية بشكل واضح ومنسق.
"""

    result = llm(prompt, max_new_tokens=200)[0]["generated_text"]

    return result

# -----------------------
# تشغيل التطبيق
# -----------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المدخلة", use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل الذكي..."):

            # 1. وصف الصورة
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            description = processor.decode(out[0], skip_special_tokens=True)

            # 2. تحليل ذكي حقيقي
            analysis = ai_fashion_analysis(description)

            # -----------------------
            # عرض النتائج
            # -----------------------
            st.subheader("📌 الوصف:")
            st.write(description)

            st.subheader("🧠 التحليل الذكي:")
            st.write(analysis)