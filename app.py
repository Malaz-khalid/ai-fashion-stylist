import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist (Stylist Mode)")
st.write("تحليل + اقتراحات لتحسين اللبس")

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
# تحليل + اقتراحات ذكية
# -----------------------
def fashion_stylist(description):

    text = description.lower()

    score = 5
    style = "كاجوال"
    points = []
    improvements = []
    outfit_tips = []

    # -------- beach / summer --------
    if any(w in text for w in ["beach", "ocean", "sea"]):
        style = "ستايل شاطئ"
        score += 2
        points.append("إطلالة مناسبة للأجواء الصيفية")

        outfit_tips.append("👗 جرّب إضافة كيمونو خفيف أو شورت صيفي")
        outfit_tips.append("🕶️ نظارة شمسية ستكمل اللوك بشكل ممتاز")
        outfit_tips.append("👡 صندل مفتوح أفضل من الأحذية الثقيلة")

    # -------- casual --------
    if "jeans" in text or "walking" in text:
        style = "كاجوال"
        points.append("إطلالة يومية مريحة")

        outfit_tips.append("🧥 إضافة جاكيت خفيف يرفع الستايل")
        outfit_tips.append("⌚ ساعة يد بسيطة تعطي لمسة أنيقة")

    # -------- colors --------
    if any(w in text for w in ["black", "white"]):
        score += 1
        points.append("ألوان كلاسيكية")

        outfit_tips.append("🎨 جربي إضافة لون واحد قوي مثل الأحمر أو الأزرق")

    # -------- dress --------
    if "dress" in text:
        style = "إطلالة أنثوية أنيقة"
        score += 2
        points.append("فستان أنيق ومميز")

        outfit_tips.append("💎 حزام خصر سيحسن شكل الفستان")
        outfit_tips.append("👜 حقيبة صغيرة تعطي لمسة فخامة")

    # -------- fallback --------
    if not outfit_tips:
        outfit_tips.append("✨ إضافة إكسسوارات بسيطة ستطور اللوك")
        outfit_tips.append("🎯 تحسين تناسق الألوان سيجعل الإطلالة أقوى")

    return f"""
التقييم: {score}/10

الستايل: {style}

تحليل الإطلالة:
{chr(10).join("✔ " + p for p in points)}

اقتراحات لتحسين اللوك:
{chr(10).join(outfit_tips)}

الانطباع العام:
الإطلالة جيدة ويمكن تطويرها لتصبح أكثر تميزاً وأناقة
"""

# -----------------------
# رفع الصورة
# -----------------------
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    if st.button("تحليل + اقتراحات"):

        with st.spinner("جاري تحليل الستايل..."):

            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            description = processor.decode(out[0], skip_special_tokens=True)

            result = fashion_stylist(description)

            st.subheader("📌 وصف الصورة:")
            st.write(description)

            st.subheader("🧠 تحليل + اقتراحات ستايل:")
            st.write(result)