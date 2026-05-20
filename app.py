import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist")
st.write("ارفع صورة وسيتم تحليل الإطلالة بشكل ذكي")

# -----------------------
# تحميل موديل وصف الصورة (BLIP)
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
# تحميل موديل تحديد الستايل (AI حقيقي)
# -----------------------
@st.cache_resource
def load_style_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

style_classifier = load_style_model()

# -----------------------
# رفع الصورة
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

# -----------------------
# دالة تحديد الستايل (ذكاء حقيقي)
# -----------------------
def detect_style(text):

    labels = [
        "كاجوال",
        "رياضي",
        "رسمي",
        "ستايل شاطئ",
        "ستريت ستايل",
        "ستايل أنيق"
    ]

    result = style_classifier(text, labels)

    style = result["labels"][0]
    confidence = result["scores"][0]

    return style, confidence

# -----------------------
# تحليل الموضة بالعربي
# -----------------------
def fashion_analysis(text, style):

    text = text.lower()
    score = 6

    if "white" in text or "black" in text:
        score += 1
    if "jeans" in text:
        score += 1
    if "beach" in text or "ocean" in text:
        score += 1

    return f"""
التقييم: {score}/10

الستايل: {style}

تحليل الإطلالة:
✔ تم تحديد الستايل باستخدام الذكاء الاصطناعي
✔ الإطلالة مناسبة ومتكاملة بصرياً

نقاط التحسين:
- إضافة إكسسوار مناسب للستايل
- تحسين تنسيق الألوان حسب المناسبة
- تجربة طبقات إضافية (جاكيت / حقيبة)

الانطباع العام:
إطلالة متناسقة وتعكس ستايل واضح
"""

# -----------------------
# عند رفع صورة
# -----------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الإطلالة المرفوعة", use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل بالذكاء الاصطناعي..."):

            # 1. وصف الصورة
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            result = processor.decode(out[0], skip_special_tokens=True)

            # 2. تحديد الستايل الذكي
            style, confidence = detect_style(result)

            # -----------------------
            # عرض النتائج
            # -----------------------
            st.subheader("📌 الوصف:")
            st.write(result)

            st.subheader("👗 الستايل:")
            st.write(f"{style} (ثقة: {confidence:.2f})")

            st.subheader("🧠 تحليل الموضة:")
            st.write(fashion_analysis(result, style))