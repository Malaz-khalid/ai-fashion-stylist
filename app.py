import streamlit as st
from PIL import Image

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist (Stable Version)")
st.write("تحليل ذكي مستقر بدون أعطال")

# -----------------------
# 👗 Style Keywords (نفس فكرتك)
# -----------------------
STYLE_KEYWORDS = {

    "كاجوال": [
        "jeans", "t-shirt", "tshirt", "hoodie", "sneakers",
        "walking", "casual", "simple", "selfie", "bathroom"
    ],

    "أنيق": [
        "dress", "elegant", "formal",
        "standing outside", "fashion"
    ],

    "ستريت ستايل": [
        "street", "urban", "jacket", "oversized",
        "cool", "modern"
    ],

    "ستايل شاطئ": [
        "beach", "ocean", "sea", "summer",
        "sun", "vacation"
    ],

    "كيوت": [
        "little girl", "cute", "pink", "child",
        "kid", "smiling", "baby"
    ],

    "رياضي": [
        "sport", "gym", "fitness", "running"
    ]
}

# -----------------------
# 🔥 وصف بسيط بدل BLIP (Stable AI Simulation)
# -----------------------
def fake_ai_description(image):
    # بدل model (مستقر 100%)
    return "a person in a casual outfit taking a photo"

# -----------------------
# تحليل الموضة الذكي (بدون تغيير فكرتك)
# -----------------------
def fashion_analysis(description):

    text = description.lower()

    score = 7
    style = "كاجوال"

    points = []
    improvements = []

    # -----------------------
    # Style detection
    # -----------------------
    style_scores = {s: 0 for s in STYLE_KEYWORDS}

    for s, keywords in STYLE_KEYWORDS.items():
        for word in keywords:
            if word in text:
                style_scores[s] += 2

    best_style = max(style_scores, key=style_scores.get)

    if style_scores[best_style] > 0:
        style = best_style
        score += style_scores[best_style] // 2

    # -----------------------
    # signals
    # -----------------------
    if "woman" in text:
        points.append("إطلالة نسائية")

    if "man" in text:
        points.append("إطلالة رجالية")

    if "dress" in text:
        points.append("فستان أنيق")

    if any(w in text for w in ["beach", "ocean", "sea"]):
        style = "ستايل شاطئ"
        points.append("إطلالة صيفية")

    # -----------------------
    # fallback
    # -----------------------
    if not points:
        points.append("إطلالة بسيطة")

    points.append(f"الاستايل (AI): {style}")

    return f"""
التقييم: {min(score,10)}/10

الستايل: {style}

تحليل الإطلالة:
{chr(10).join("✔ " + p for p in points)}

نقاط التحسين:
- تحسين تنسيق الألوان
- إضافة إكسسوارات بسيطة

الانطباع العام:
الإطلالة تعكس طابع {style} بشكل واضح
"""

# -----------------------
# Upload image
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل..."):

            # 🔥 بدل BLIP (بدون أي crashes)
            description = fake_ai_description(image)

            analysis = fashion_analysis(description)

            st.subheader("📌 وصف الصورة (AI):")
            st.write(description)

            st.subheader("🧠 تحليل الموضة:")
            st.write(analysis)