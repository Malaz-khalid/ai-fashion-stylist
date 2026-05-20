import streamlit as st
from PIL import Image
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------
# UI احترافي
# -----------------------
st.set_page_config(
    page_title="AI Fashion Stylist",
    page_icon="🖤",
    layout="centered"
)

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #111;
}
.sub-text {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.block {
    background-color: #f7f7f7;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🖤 AI Fashion Stylist</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>تحليل + اقتراح Outfit + ذكاء بصري</div>", unsafe_allow_html=True)

# -----------------------
# Model
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
# Color detection
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
# Scene
# -----------------------
def detect_scene(text):
    text = text.lower()

    if any(w in text for w in ["beach", "ocean", "sea"]):
        return "شاطئ"
    elif any(w in text for w in ["street", "walking"]):
        return "شارع"
    return "عام"

# -----------------------
# Outfit Suggestions (جديد 🔥)
# -----------------------
def outfit_recommendation(style, colors, scene):

    outfits = []

    if "شاطئ" in scene:
        outfits = [
            "👗 فستان صيفي خفيف + صندل",
            "🩳 شورت + تيشيرت قطني + قبعة",
            "🕶️ نظارة شمسية + كيمونو خفيف"
        ]

    elif "شارع" in scene:
        outfits = [
            "👖 جينز + تيشيرت + جاكيت خفيف",
            "👟 سنيكرز + هودي كاجوال",
            "🧥 بليزر كاجوال + بنطلون واسع"
        ]

    else:
        outfits = [
            "👕 تيشيرت بسيط + جينز",
            "👗 فستان أو قميص كاجوال",
            "🧢 إطلالة بسيطة بإكسسوارات خفيفة"
        ]

    # تعديل حسب الألوان
    if "داكنة" in colors:
        outfits.append("🖤 ألوان غامقة لإطلالة قوية وفخمة")

    if "فاتحة" in colors:
        outfits.append("🤍 ألوان فاتحة لإطلالة ناعمة ومريحة")

    return outfits

# -----------------------
# Analysis
# -----------------------
def fashion_analysis(description, colors, scene):

    text = description.lower()

    score = 5
    style = "كاجوال"
    points = []

    if "beach" in text:
        style = "ستايل شاطئ"
        score += 2
        points.append("مناسب للأجواء الصيفية")

    if "street" in text:
        style = "ستريت ستايل"
        score += 1
        points.append("إطلالة يومية عصرية")

    if "dress" in text:
        style = "إطلالة أنيقة"
        score += 2
        points.append("فستان أنيق")

    if not points:
        points.append("إطلالة بسيطة")

    return score, style, points

# -----------------------
# Upload
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل..."):

            # AI caption
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            description = processor.decode(out[0], skip_special_tokens=True)

            # signals
            colors = detect_colors(image)
            scene = detect_scene(description)

            score, style, points = fashion_analysis(description, colors, scene)

            outfits = outfit_recommendation(style, colors, scene)

            # -----------------------
            # UI Output
            # -----------------------
            st.markdown("## 📌 الوصف")
            st.markdown(f"<div class='block'>{description}</div>", unsafe_allow_html=True)

            st.markdown("## 🧠 التحليل")
            st.markdown(f"<div class='block'>التقييم: {score}/10<br>الستايل: {style}<br>{chr(10).join(points)}</div>", unsafe_allow_html=True)

            st.markdown("## 👗 اقتراحات Outfit")

            for o in outfits:
                st.markdown(f"- {o}")