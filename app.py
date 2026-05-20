import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist (Hybrid AI)")
st.write("تحليل ذكي يجمع بين AI + منطق الموضة")

# -----------------------
# BLIP Model
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
# تحليل الموضة الذكي
# -----------------------
def fashion_analysis(description):

    text = description.lower()

    score = 5
    style = "كاجوال"
    points = []
    improvements = []

    # -------- AI signals --------
    if any(w in text for w in ["beach", "ocean", "sea"]):
        style = "ستايل شاطئ"
        score += 2
        points.append("إطلالة مناسبة للأجواء الساحلية")
        improvements.append("إضافة نظارة شمسية أو قبعة صيفية")

    # -------- gender signals (معدل بشكل صحيح) --------
    if "woman" in text and "dress" in text:
        points.append("إطلالة نسائية أنيقة")
    elif "woman" in text:
        points.append("إطلالة نسائية")

    elif "man" in text and "suit" in text:
        points.append("إطلالة رجالية رسمية")
    elif "man" in text:
        points.append("إطلالة رجالية")

    # -------- street style --------
    if any(w in text for w in ["street", "walking"]):
        style = "ستريت ستايل"
        score += 1
        points.append("إطلالة يومية مناسبة للخروج")
        improvements.append("إضافة حقيبة أو جاكيت خفيف")

    # -------- colors --------
    if any(w in text for w in ["black", "white"]):
        score += 1
        points.append("ألوان كلاسيكية أنيقة")
        improvements.append("يمكن إضافة لون مميز لكسر الحيادية")

    # -------- dress --------
    if "dress" in text:
        style = "إطلالة أنيقة"
        score += 2
        points.append("فستان أنيق ومميز")
        improvements.append("إضافة إكسسوارات بسيطة لرفع اللوك")

    # -------- fallback --------
    if not points:
        points.append("إطلالة بسيطة ومتوازنة")
        improvements.append("تحسين تنسيق الألوان والإكسسوارات")

    return f"""
التقييم: {score}/10

الستايل: {style}

تحليل الإطلالة:
{chr(10).join("✔ " + p for p in points)}

نقاط التحسين:
{chr(10).join("- " + i for i in improvements)}

الانطباع العام:
الإطلالة تعكس طابع {style} بشكل واضح مع إمكانية تطويرها لتصبح أكثر تميزاً
"""

# -----------------------
# Upload image
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل الذكي..."):

            # 1. AI caption
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            description = processor.decode(out[0], skip_special_tokens=True)

            # 2. Hybrid analysis
            analysis = fashion_analysis(description)

            # -----------------------
            # النتائج
            # -----------------------
            st.subheader("📌 وصف الصورة (AI):")
            st.write(description)

            st.subheader("🧠 تحليل الموضة:")
            st.write(analysis)