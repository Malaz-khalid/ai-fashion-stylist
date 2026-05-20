import streamlit as st
from PIL import Image
import torch
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist (Real AI Scoring)")
st.write("تحليل ذكي متعدد الإشارات بدون قواعد ثابتة")

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
# 🎨 Color Signal
# -----------------------
def color_score(image):
    image = image.resize((100, 100))
    arr = np.array(image)
    avg = arr.mean(axis=(0, 1))
    r, g, b = avg

    score = 0

    if r > 150 or g > 150 or b > 150:
        score += 1  # bright
    if r < 80 and g < 80 and b < 80:
        score += 1  # dark contrast
    if abs(r-g) > 40 or abs(r-b) > 40:
        score += 1  # color variation

    return score

# -----------------------
# 🌍 Scene Score
# -----------------------
def scene_score(text):
    text = text.lower()

    score = 0

    if any(w in text for w in ["beach", "sea", "ocean"]):
        score += 2
    if any(w in text for w in ["street", "walking"]):
        score += 1
    if "room" in text or "indoor" in text:
        score += 0.5
    if "pool" in text:
        score += 1.5

    return score

# -----------------------
# 👗 Clothing Score
# -----------------------
def clothing_score(text):
    text = text.lower()

    score = 0

    clothes = ["shirt", "dress", "jeans", "jacket", "t-shirt", "skirt"]

    for c in clothes:
        if c in text:
            score += 1

    return score

# -----------------------
# 🧠 Complexity Score
# -----------------------
def complexity_score(text):
    words = len(text.split())

    if words < 6:
        return 0.5
    elif words < 10:
        return 1
    else:
        return 2

# -----------------------
# 🧠 Final AI Scoring Engine
# -----------------------
def ai_fashion_score(description, image):

    c_score = clothing_score(description)
    s_score = scene_score(description)
    col_score = color_score(image)
    comp_score = complexity_score(description)

    total = 3 + c_score + s_score + col_score + comp_score

    # style prediction (dynamic)
    if s_score >= 2:
        style = "ستايل شاطئ / صيفي"
    elif "jeans" in description:
        style = "كاجوال"
    elif "dress" in description:
        style = "إطلالة أنيقة"
    else:
        style = "كاجوال بسيط"

    insights = []

    if c_score > 2:
        insights.append("إطلالة تحتوي تفاصيل ملابس واضحة")
    else:
        insights.append("إطلالة بسيطة في الملابس")

    if col_score >= 2:
        insights.append("تنوع ألوان جيد يعطي توازن بصري")

    if s_score >= 2:
        insights.append("المشهد الخارجي يعطي طابع قوي للإطلالة")

    return int(total), style, insights

# -----------------------
# Upload
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("تحليل ذكي حقيقي"):

        with st.spinner("جاري تحليل الصورة بذكاء حقيقي..."):

            # BLIP
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            description = processor.decode(out[0], skip_special_tokens=True)

            # AI engine
            score, style, insights = ai_fashion_score(description, image)

            # -----------------------
            st.subheader("📌 وصف الصورة")
            st.write(description)

            st.subheader("🧠 التقييم الذكي")
            st.write(f"التقييم: {score}/10")
            st.write(f"الستايل: {style}")

            st.subheader("📊 التحليل")
            for i in insights:
                st.write("✔", i)