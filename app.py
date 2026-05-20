import streamlit as st
from PIL import Image
import torch
import sqlite3
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
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model

processor, model = load_model()

# -----------------------
# Database (حفظ النتائج داخلياً)
# -----------------------
conn = sqlite3.connect("fashion_ai.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT,
    description TEXT,
    style TEXT,
    score INTEGER
)
""")

conn.commit()

# -----------------------
# حفظ النتائج
# -----------------------
def save_result(image_name, description, style, score):

    cursor.execute("""
    INSERT INTO results (
        image_name,
        description,
        style,
        score
    )
    VALUES (?, ?, ?, ?)
    """, (image_name, description, style, score))

    conn.commit()

# -----------------------
# 👗 Style Keywords
# -----------------------
STYLE_KEYWORDS = {

    "كاجوال": [
        "jeans",
        "t-shirt",
        "tshirt",
        "hoodie",
        "sneakers",
        "walking",
        "casual",
        "simple",
        "selfie",
        "bathroom",
        "striped shirt",
        "streetwear",
        "relaxed",
        "daily",
        "everyday"
    ],

    "أنيق": [
        "dress",
        "white dress",
        "maroon dress",
        "elegant",
        "formal",
        "fashion",
        "luxury",
        "classic",
        "blazer",
        "heels",
        "gown",
        "stylish"
    ],

    "ستريت ستايل": [
        "street",
        "urban",
        "jacket",
        "oversized",
        "modern",
        "trend",
        "cool",
        "baggy",
        "fashionable"
    ],

    "ستايل شاطئ": [
        "beach",
        "ocean",
        "sea",
        "summer",
        "sun",
        "vacation",
        "sand",
        "pool",
        "tropical",
        "outdoor"
    ],

    "رياضي": [
        "sport",
        "gym",
        "fitness",
        "running",
        "training",
        "active",
        "tracksuit",
        "leggings"
    ],

    "محجبات / محتشم": [
        "head scarf",
        "hijab",
        "modest",
        "abaya",
        "long dress"
    ]
}

# -----------------------
# تحليل الموضة الذكي
# -----------------------
def fashion_analysis(description):

    text = description.lower()

    score = 5
    style = "كاجوال"

    points = []
    improvements = []

    # -----------------------
    # Style Scoring
    # -----------------------
    style_scores = {
        s: 0 for s in STYLE_KEYWORDS
    }

    for s, keywords in STYLE_KEYWORDS.items():

        for word in keywords:

            if word in text:
                style_scores[s] += 2

    # -----------------------
    # Smart bonus scoring
    # -----------------------
    if "jeans" in text and "t-shirt" in text:
        style_scores["كاجوال"] += 3

    if "dress" in text and "white" in text:
        style_scores["أنيق"] += 3

    if "beach" in text or "ocean" in text:
        style_scores["ستايل شاطئ"] += 3

    if "selfie" in text and "bathroom" in text:
        style_scores["كاجوال"] += 2

    if "head scarf" in text or "hijab" in text:
        style_scores["محجبات / محتشم"] += 3

    # -----------------------
    # تحديد أعلى ستايل
    # -----------------------
    best_style = max(style_scores, key=style_scores.get)

    if style_scores[best_style] > 0:
        style = best_style
        score += style_scores[best_style] // 2

    # -----------------------
    # Scene analysis
    # -----------------------
    if any(w in text for w in ["beach", "ocean", "sea"]):

        points.append("إطلالة مناسبة للأجواء الساحلية")
        improvements.append("إضافة نظارة شمسية أو قبعة صيفية")

    if any(w in text for w in ["store", "display"]):

        points.append("الصورة تبدو كعرض أزياء أو متجر")
        improvements.append("تحسين الإضاءة لإظهار تفاصيل القطعة")

    if any(w in text for w in ["bathroom", "mirror", "selfie"]):

        points.append("إطلالة عفوية مناسبة للسوشيال ميديا")
        improvements.append("تحسين الخلفية أو زاوية التصوير")

    # -----------------------
    # Gender analysis
    # -----------------------
    if "woman" in text and "dress" in text:
        points.append("إطلالة نسائية أنيقة")

    elif "woman" in text:
        points.append("إطلالة نسائية")

    elif "man" in text and "suit" in text:
        points.append("إطلالة رجالية رسمية")

    elif "man" in text:
        points.append("إطلالة رجالية")

    # -----------------------
    # Color analysis
    # -----------------------
    if "black" in text:
        points.append("اللون الأسود يعطي طابع أنيق")

    if "white" in text:
        points.append("اللون الأبيض يعطي إحساس بالنظافة والبساطة")

    if "red" in text:
        points.append("اللون الأحمر يضيف جرأة وطاقة للإطلالة")

    if "green" in text:
        points.append("اللون الأخضر يعطي إحساس بالحيوية")

    if "yellow" in text:
        points.append("اللون الأصفر يضيف لمسة مشرقة")

    # -----------------------
    # Clothing analysis
    # -----------------------
    if "dress" in text:
        points.append("الفستان عنصر أساسي في الإطلالة")

    if "jeans" in text:
        points.append("الجينز يضيف طابع كاجوال عملي")

    if "jacket" in text:
        points.append("الجاكيت يعزز الطبقات والأسلوب")

    if "hoodie" in text:
        points.append("الهودي يعطي طابع شبابي مريح")

    # -----------------------
    # Improvements
    # -----------------------
    if style == "كاجوال":
        improvements.append("إضافة إكسسوارات بسيطة لرفع اللوك")

    elif style == "أنيق":
        improvements.append("تنسيق حقيبة أو حذاء أكثر فخامة")

    elif style == "ستريت ستايل":
        improvements.append("استخدام ألوان جريئة أكثر")

    elif style == "رياضي":
        improvements.append("تنسيق الحذاء الرياضي مع الإطلالة")

    # -----------------------
    # fallback
    # -----------------------
    if not points:

        points.append("إطلالة بسيطة ومتوازنة")
        improvements.append("تحسين تنسيق الألوان والإكسسوارات")

    # -----------------------
    # تحليل نهائي
    # -----------------------
    points.append(f"الاستايل المتوقع (AI): {style}")

    analysis_text = f"""
التقييم: {min(score,10)}/10

الستايل: {style}

تحليل الإطلالة:
{chr(10).join("✔ " + p for p in points)}

نقاط التحسين:
{chr(10).join("- " + i for i in improvements)}

الانطباع العام:
الإطلالة تعكس طابع {style} بشكل واضح مع تحليل ذكي متعدد المصادر
"""

    return analysis_text, style, score

# -----------------------
# Upload image
# -----------------------
uploaded_file = st.file_uploader(
    "ارفع صورة",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل الذكي..."):

            # -----------------------
            # AI Caption
            # -----------------------
            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():

                out = model.generate(
                    **inputs,
                    max_new_tokens=50
                )

            description = processor.decode(
                out[0],
                skip_special_tokens=True
            )

            # -----------------------
            # Fashion Analysis
            # -----------------------
            analysis, style, score = fashion_analysis(description)

            # -----------------------
            # Save Results
            # -----------------------
            save_result(
                uploaded_file.name,
                description,
                style,
                score
            )

            # -----------------------
            # النتائج
            # -----------------------
            st.subheader("📌 وصف الصورة (AI):")
            st.write(description)

            st.subheader("🧠 تحليل الموضة:")
            st.write(analysis)