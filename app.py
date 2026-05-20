import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------
# إعداد الصفحة
# -----------------------
st.set_page_config(
    page_title="AI Fashion Stylist",
    layout="centered"
)

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
        "bathroom"
    ],

    "أنيق": [
        "dress",
        "white dress",
        "elegant",
        "formal",
        "standing outside",
        "fashion",
        "maroon dress"
    ],

    "ستريت ستايل": [
        "street",
        "urban",
        "jacket",
        "oversized",
        "cool",
        "modern",
        "trend"
    ],

    "ستايل شاطئ": [
        "beach",
        "ocean",
        "sea",
        "summer",
        "sun",
        "vacation",
        "outdoor"
    ],

    "كيوت": [
        "little girl",
        "cute",
        "pink",
        "child",
        "kid",
        "smiling",
        "baby",
        "soft colors",
        "little boy"
    ],

    "رياضي": [
        "sport",
        "gym",
        "fitness",
        "running"
    ]
}

# -----------------------
# تحليل الموضة الذكي
# -----------------------
def fashion_analysis(description):

    text = description.lower()

    score = 7
    style = "كاجوال"

    points = []
    improvements = []

    cute_mode = False

    # -----------------------
    # 🔥 NEW: Style Detection
    # -----------------------
    style_scores = {
        s: 0 for s in STYLE_KEYWORDS
    }

    for s, keywords in STYLE_KEYWORDS.items():

        for word in keywords:

            if word in text:
                style_scores[s] += 2

    # -----------------------
    # Cute style detection
    # -----------------------
    if "little girl" in text or "little boy" in text:

        style = "كيوت"

        score = 10

        cute_mode = True

        points.append(
            "إطلالة كيوت طفولية لطيفة"
        )

        improvements.append(
            "إضافة ألوان مرحة أو إكسسوارات ناعمة"
        )

        style_scores["كيوت"] += 100

    # -----------------------
    # تحديد أفضل ستايل
    # -----------------------
    best_style = max(
        style_scores,
        key=style_scores.get
    )

    if style_scores[best_style] > 0 and not cute_mode:

        style = best_style

        score += style_scores[best_style] // 2

    # -----------------------
    # AI signals
    # -----------------------
    if any(w in text for w in ["beach", "ocean", "sea"]):

        style = "ستايل شاطئ"

        score += 2

        points.append(
            "إطلالة مناسبة للأجواء الساحلية"
        )

        improvements.append(
            "إضافة نظارة شمسية أو قبعة صيفية"
        )

    # -----------------------
    # gender signals
    # -----------------------
    if "woman" in text and "dress" in text:

        points.append(
            "إطلالة نسائية أنيقة"
        )

    elif "woman" in text:

        points.append(
            "إطلالة نسائية"
        )

    elif "man" in text and "suit" in text:

        points.append(
            "إطلالة رجالية رسمية"
        )

    elif "man" in text:

        points.append(
            "إطلالة رجالية"
        )

    # -----------------------
    # street style
    # -----------------------
    if any(w in text for w in ["street", "walking"]):

        style = "ستريت ستايل"

        score += 2

        points.append(
            "إطلالة يومية مناسبة للخروج"
        )

        improvements.append(
            "إضافة حقيبة أو جاكيت خفيف"
        )

    # -----------------------
    # colors
    # -----------------------
    if any(w in text for w in ["black", "white"]):

        score += 2

        points.append(
            "ألوان كلاسيكية أنيقة"
        )

        improvements.append(
            "يمكن إضافة لون مميز لكسر الحيادية"
        )

    if "pink" in text:

        points.append(
            "اللون الوردي يعزز الطابع الكيوت"
        )

    if "red" in text:

        points.append(
            "اللون الأحمر يضيف جرأة للإطلالة"
        )

    if "green" in text:

        points.append(
            "اللون الأخضر يعطي حيوية وانتعاش"
        )

    # -----------------------
    # dress
    # -----------------------
    if "dress" in text and not cute_mode:

        style = "إطلالة أنيقة"

        score += 5

        points.append(
            "فستان أنيق ومميز"
        )

        improvements.append(
            "إطلالة رائعة ومتناسقة"
        )

    # -----------------------
    # hoodie
    # -----------------------
    if "hoodie" in text:

        points.append(
            "الهودي يعطي ستايل شبابي عصري"
        )

    # -----------------------
    # jeans
    # -----------------------
    if "jeans" in text:

        points.append(
            "الجينز يعطي طابع كاجوال عملي"
        )

    # -----------------------
    # fallback
    # -----------------------
    if not points:

        points.append(
            "إطلالة بسيطة ومتوازنة"
        )

        improvements.append(
            "إضافة إكسسوارات لإكمال اللوك"
        )

    # -----------------------
    # Final AI Style
    # -----------------------
    points.append(
        f"الاستايل (AI): {style}"
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
# Upload image
# -----------------------
uploaded_file = st.file_uploader(
    "ارفع صورة",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        use_container_width=True
    )

    if st.button("تحليل الإطلالة"):

        with st.spinner(
            "جاري التحليل الذكي..."
        ):

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
                    max_new_tokens=20
                )

            description = processor.decode(
                out[0],
                skip_special_tokens=True
            )

            # -----------------------
            # تنظيف الذاكرة
            # -----------------------
            torch.cuda.empty_cache()

            # -----------------------
            # Hybrid Analysis
            # -----------------------
            analysis = fashion_analysis(
                description
            )

            # -----------------------
            # النتائج
            # -----------------------
            st.subheader(
                "📌 وصف الصورة (AI):"
            )

            st.write(description)

            st.subheader(
                "🧠 تحليل الموضة:"
            )

            st.write(analysis)