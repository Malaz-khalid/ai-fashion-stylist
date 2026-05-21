import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(page_title="AI Fashion Stylist", layout="centered")

st.title("🖤 AI Fashion Stylist ")
st.write("تحليل استايلك")

# -----------------------
# تحميل النموذج (محمي من التكرار)
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
# Style Keywords
# -----------------------
STYLE_KEYWORDS = {

    "كاجوال": [
        "jeans",
        "t-shirt",
        "hoodie",
        "sneakers",
        "casual",
        "selfie",
        "bathroom",
        "mirror",
        "daily",
        "simple",
        "relaxed",
        "comfortable",
        "shirt",
        "pants"
    ],

    "فورمال": [
        "dress",
        "white dress",
        "gown",
        "fashion",
        "formal",
        "luxury",
        "heels",
        "elegant",
        "stylish",
        "chic",
        "beautiful",
        "party",
        "wedding",
        "makeup",
        "jewelry",
        "suit",
        "tie",
        "blazer"
    ],

    "ستريت ستايل": [
        "street",
        "urban",
        "oversized",
        "cool",
        "modern",
        "jacket",
        "baggy",
        "fashion week",
        "hoodie",
        "denim",
        "sneaker",
        "city",
        "trendy"
    ],

    "ستايل شاطئ": [
        "beach",
        "ocean",
        "sea",
        "summer",
        "vacation",
        "sun",
        "sand",
        "resort",
        "island",
        "outdoor",
        "coast"
    ],

    "كيوت": [
        "little girl",
        "child",
        "kid",
        "baby",
        "cute",
        "pink",
        "smiling",
        "soft colors",
        "toy",
        "small",
        "happy",
        "playful",
        "bow",
        "cartoon",
        "little boy"
    ],

    "رياضي": [
        "sport",
        "gym",
        "fitness",
        "running",
        "workout",
        "training",
        "athletic",
        "sportswear",
        "jogging",
        "active",
        "nike",
        "adidas"
    ],

    "محجبات": [
        "hijab",
        "head scarf",
        "abaya",
        "modest",
        "veil",
        "long dress",
        "covered",
        "muslim fashion"
    ],

    "شتوي": [
        "coat",
        "winter",
        "jacket",
        "boots",
        "scarf",
        "cold",
        "sweater",
        "knit",
        "layered"
    ],

    "صيفي": [
        "shorts",
        "summer",
        "light colors",
        "sleeveless",
        "sunny",
        "tank top",
        "floral",
        "bright"
    ]
}

# -----------------------
# 🔥 تحليل الموضة
# -----------------------
def fashion_analysis(text):

    text = text.lower()

    score = 8
    style = "كاجوال"

    points = []
    improvements = []

    # -----------------------
    # style detection
    # -----------------------
    style_scores = {s: 0 for s in STYLE_KEYWORDS}

    for s, keywords in STYLE_KEYWORDS.items():

        for w in keywords:

            if w in text:
                style_scores[s] += 2

    # -----------------------
    # Cute detection
    # -----------------------
    if "little girl" in text:

        style_scores["كيوت"] += 100

        score = 10

        points.append(
            "إطلالة كيوت طفولية لطيفة"
        )

        improvements.append(
            "لوك جميل ولطيف"
        )

    # -----------------------
    # Beach style
    # -----------------------
    if any(w in text for w in ["beach", "ocean", "sea"]):

        style_scores["ستايل شاطئ"] += 4

        score += 2

        points.append(
            "إطلالة مناسبة للأجواء الساحلية"
        )

        improvements.append(
            "إضافة نظارة شمسية أو قبعة صيفية"
        )

    # -----------------------
    # Street style
    # -----------------------
    if any(w in text for w in ["street", "walking"]):

        style_scores["ستريت ستايل"] += 4

        score += 2

        points.append(
            "إطلالة يومية مناسبة للخروج"
        )

        improvements.append(
            "إضافة نظارة"
        )

    # -----------------------
    # Elegant style
    # -----------------------
    if "dress" in text:

        style_scores["فورمال"] += 5

        score += 3

        points.append(
            "فستان أنيق ومميز"
        )

        improvements.append(
            "إطلالة بنوتة راقية وهادية وواثقة"
        )

    # -----------------------
    # Colors
    # -----------------------
    if any(w in text for w in ["black", "white"]):

        score += 2

        points.append(
            "ألوان كلاسيكية أنيقة"
        )

        improvements.append(
            "إضافة لون قوي لكسر الحيادية"
        )

    # -----------------------
    # More color analysis
    # -----------------------

    # الأحمر
    if "red" in text:

        score += 2

        points.append(
            "اللون الأحمر يعطي الإطلالة طاقة وجاذبية"
        )

        improvements.append(
            "تنسيق الإكسسوارات بألوان هادئة"
        )

    # الأزرق
    if "blue" in text:

        score += 2

        points.append(
            "الأزرق يعكس هدوء وأناقة"
        )

        improvements.append(
            "إضافة تفاصيل عصرية لزيادة التميز"
        )

    # الوردي
    if "pink" in text:

        score += 2

        style_scores["كيوت"] += 4

        points.append(
            "الوردي يعطي طابع كيوت وناعم"
        )

        improvements.append(
            "إضافة أكسسوار بسيط وناعم"
        )

    # الأخضر
    if "green" in text:

        score += 2

        points.append(
            "الأخضر يعطي إحساس بالانتعاش والطبيعة"
        )

        improvements.append(
            "إضافة درجات فاتحة لزيادة الحيوية"
        )

    # الأصفر
    if "yellow" in text:

        score += 2

        points.append(
            "الأصفر يعطي طاقة وإشراقة"
        )

        improvements.append(
            "موازنة الألوان بإكسسوارات هادئة"
        )

    # البنفسجي
    if "purple" in text:

        score += 2

        points.append(
            "البنفسجي يعكس الفخامة والغموض"
        )

        improvements.append(
            "إضافة تفاصيل أنيقة بسيطة"
        )

    # البني
    if "brown" in text:

        score += 1

        points.append(
            "البني يعطي طابع دافئ وكلاسيكي"
        )

        improvements.append(
            "إضافة لون فاتح لكسر الهدوء"
        )

    # الرمادي
    if "gray" in text or "grey" in text:

        score += 1

        points.append(
            "الرمادي يعكس أناقة هادئة"
        )

        improvements.append(
            "إضافة لون قوي لإبراز الإطلالة"
        )

    # البرتقالي
    if "orange" in text:

        score += 2

        points.append(
            "البرتقالي يعطي روح مرحة وعصرية"
        )

        improvements.append(
            "تنسيق الألوان بشكل متوازن"
        )

    # -----------------------
    # gender hints
    # -----------------------
    if "woman" in text:

        points.append(
            "إطلالة نسائية جميلة"
        )

    elif "man" in text:

        points.append(
            "إطلالة رجالية أنيقة"
        )

    elif "girl" in text:

        points.append(
            "إطلالة بنوتة كيوتة وامورة"
        )

        style_scores["كيوت"] += 50

    elif "boy" in text:

        points.append(
            "إطلالة كيوت ، بيبي"
        )

        style_scores["كيوت"] += 40

    # -----------------------
    # Men's style analysis
    # -----------------------
    if "man" in text:

        # كاجوال رجالي
        if any(w in text for w in [
            "jeans",
            "t-shirt",
            "hoodie",
            "sneakers",
            "shirt"
        ]):

            style_scores["كاجوال"] += 5

            points.append(
                "إطلالة رجالية كاجوال عصرية"
            )

            improvements.append(
                "إضافة ساعة أو حذاء مميز"
            )

        # رسمي
        if any(w in text for w in [
            "suit",
            "tie",
            "formal",
            "blazer"
        ]):

            style_scores["فورمال"] += 6

            points.append(
                "إطلالة رجالية رسمية وفخمة"
            )

            improvements.append(
                "تنسيق الحذاء مع الإطلالة"
            )

        # رياضي
        if any(w in text for w in [
            "gym",
            "sport",
            "fitness",
            "running",
            "sportswear"
        ]):

            style_scores["رياضي"] += 6

            points.append(
                "إطلالة رياضية شبابية"
            )

            improvements.append(
                "إضافة ألوان رياضية أكثر حيوية"
            )

        # ستريت ستايل
        if any(w in text for w in [
            "street",
            "urban",
            "oversized",
            "jacket",
            "hoodie"
        ]):

            style_scores["ستريت ستايل"] += 6

            points.append(
                "إطلالة ستريت ستايل شبابية"
            )

            improvements.append(
                "إضافة أكسسوار شبابي مميز"
            )

        # شاطئ
        if any(w in text for w in [
            "beach",
            "ocean",
            "sea"
        ]):

            style_scores["ستايل شاطئ"] += 5

            points.append(
                "إطلالة صيفية مريحة"
            )

            improvements.append(
                "إضافة نظارة شمسية عصرية"
            )

    # -----------------------
    # تحديد أفضل ستايل
    # -----------------------
    best_style = max(
        style_scores,
        key=style_scores.get
    )

    if style_scores[best_style] > 0:

        style = best_style

    # -----------------------
    # fallback
    # -----------------------
    if not points:

        points.append(
            "إطلالة عامة متوازنة"
        )

        improvements.append(
            "إضافة لمسات عصرية أكثر"
        )

    # -----------------------
    # Final AI Style
    # -----------------------
    points.append(
        f"الستايل (AI): {style}"
    )

    return f"""
التقييم: {min(score,10)}/10

الستايل: {style}

تحليل الإطلالة:
{chr(10).join("✔ " + p for p in points)}

نقاط التحسين:
{chr(10).join("- " + i for i in improvements)}

الانطباع العام:
الإطلالة تعكس طابع {style} بشكل واضح مع تحليل ذكي متعدد المصادر
"""

# -----------------------
# 🔥 آمن جداً ضد الكراش
# -----------------------
def safe_caption(image):

    try:

        image = image.convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():

            out = model.generate(
                **inputs,
                max_new_tokens=40
            )

        caption = processor.decode(
            out[0],
            skip_special_tokens=True
        )

        return caption

    except Exception as e:

        return "a person wearing casual outfit"

# -----------------------
# Upload
# -----------------------
uploaded_file = st.file_uploader(
    "ارفع صورة",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image)

    if st.button("تحليل الإطلالة"):

        with st.spinner("جاري التحليل الذكي..."):

            # AI Caption
            description = safe_caption(image)

            # Fashion Analysis
            result = fashion_analysis(description)

            st.subheader("📌 وصف الصورة (AI):")
            st.write(description)

            st.subheader("🧠 تحليل الموضة:")
            st.write(result)
