import streamlit as st
from PIL import Image
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist (Hybrid AI++)")
st.write("تحليل ذكي + ألوان + سياق + تحسينات")

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
# 🎨 Color Detection (جديد)
# -----------------------
def detect_colors(image):
    image = image.resize((100, 100))
    arr = np.array(image)

    avg = arr.mean(axis=(0, 1))
    r, g, b = avg

    colors = []

    if r > 150 and g > 150 and b > 150:
        colors.append("ألوان فاتحة")
    if r < 80 and g < 80 and b < 80:
        colors.append("ألوان داكنة")
    if r > g and r > b:
        colors.append("ألوان دافئة")
    if b > r and b > g:
        colors.append("ألوان باردة")

    return colors

# -----------------------
# 🌍 Scene Detection (جديد)
# -----------------------
def detect_scene(text):
    text = text.lower()

    if any(w in text for w in ["beach", "ocean", "sea"]):
        return "شاطئ / خارجية"
    elif any(w in text for w in ["street", "walking", "road"]):
        return "شارع / خارجية"
    else:
        return "داخلية / غير محدد"

# -----------------------
# 👗 النظام الأساسي (محافظ عليه)
# -----------------------
def fashion_analysis(description, colors, scene):

    text = description.lower()

    score = 5
    style = "كاجوال"
    points = []
    improvements = []

    # -------- rules الأساسية (نفس فكرتك) --------
    if any(w in text for w in ["beach", "sea", "ocean"]):
        style = "ستايل شاطئ"
        score += 2
        points.append("إطلالة مناسبة للأجواء الساحلية")

    if any(w in text for w in ["street", "walking"]):
        style = "ستريت ستايل"
        score += 1
        points.append("إطلالة مناسبة للخروج")

    if "dress" in text:
        style = "إطلالة أنيقة"
        score += 2
        points.append("ملابس أنثوية أنيقة")

    if any(w in text for w in ["black", "white"]):
        score += 1
        points.append("ألوان كلاسيكية")

    # -------- التحسين الجديد (الألوان + السياق) --------
    if "شاطئ" in scene:
        score += 1
        points.append("السياق الخارجي يعطي طابع صيفي")
        improvements.append("يمكن إضافة نظارة شمسية أو قبعة")

    if "داكنة" in colors:
        points.append("الألوان الداكنة تعطي فخامة")

    if "فاتحة" in colors:
        points.append("الألوان الفاتحة تعطي خفة وأناقة")

    if not points:
        points.append("إطلالة بسيطة ومتناسقة")
        improvements.append("تحسين تنسيق الألوان والإكسسوارات")

    return f"""
التقييم: {score}/10

الستايل: {style}

السياق:
- {scene}

الألوان:
- {chr(10).join(colors) if colors else "غير محدد"}

تحليل الإطلالة:
{chr(10).join("✔ " + p for p in points)}

نقاط التحسين:
{chr(10).join("- " + i for i in improvements)}

الانطباع العام:
الإطلالة تعكس طابع {style} مع تأثير واضح من الألوان والمكان مما يجعل التحليل أكثر دقة
"""

# -----------------------
# Upload
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل الذكي..."):

            # 1. BLIP description
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            description = processor.decode(out[0], skip_special_tokens=True)

            # 2. New signals
            colors = detect_colors(image)
            scene = detect_scene(description)

            # 3. Final hybrid analysis
            analysis = fashion_analysis(description, colors, scene)

            # -----------------------
            st.subheader("📌 وصف الصورة (AI):")
            st.write(description)

            st.subheader("🎨 الألوان:")
            st.write(colors)

            st.subheader("🌍 السياق:")
            st.write(scene)

            st.subheader("🧠 التحليل النهائي:")
            st.write(analysis)