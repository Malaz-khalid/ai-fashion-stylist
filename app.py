import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist")
st.write("ارفع صورة وسيتم تحليل الإطلالة تلقائياً")

# -----------------------
# تحميل الموديل
# -----------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_model()

# -----------------------
# تحليل الموضة بالعربي
# -----------------------
def fashion_analysis(text):
    text = text.lower()
    score = 6

    if "white" in text or "black" in text:
        score += 1
    if "jeans" in text:
        score += 1
    if "shirt" in text:
        style = "كاجوال بسيط"
    else:
        style = "ستايل يومي"

    return f"""
التقييم: {score}/10

الستايل: {style}

تحليل الإطلالة:
✔ إطلالة بسيطة ومناسبة للاستخدام اليومي
✔ ألوان هادئة ومتناسقة

نقاط التحسين:
- إضافة إكسسوار بسيط مثل ساعة أو نظارة
- تجربة جاكيت خفيف لإضافة ستايل
- اختيار حذاء أكثر تناسقاً

الانطباع العام:
إطلالة كاجوال مناسبة للخروج أو الجامعة
"""

# -----------------------
# رفع الصورة
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الإطلالة المرفوعة", use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل..."):

            # -----------------------
            # استخراج وصف الصورة
            # -----------------------
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            result = processor.decode(out[0], skip_special_tokens=True)

            # -----------------------
            # عرض النتائج
            # -----------------------
            st.subheader("📌 الوصف:")
            st.write(result)

            st.subheader("👗 تحليل الموضة:")
            st.write(fashion_analysis(result))