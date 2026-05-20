import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.title("🖤 AI Fashion Stylist")
st.write("ارفع صورة وسيتم تحليلها")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("ارفع صورة", type=["jpg","png","jpeg"])

def fashion_analysis(text):
    text = text.lower()
    score = 6

    if "white" in text or "black" in text:
        score += 1
    if "jeans" in text:
        score += 1

    return f"""
التقييم: {score}/10

الستايل: كاجوال بسيط

تحليل الإطلالة:
✔ إطلالة مناسبة وبسيطة

نصائح:
- إضافة إكسسوار بسيط
- تحسين تنسيق الألوان
"""

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("تحليل"):
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        result = processor.decode(out[0], skip_special_tokens=True)

        st.subheader("الوصف:")
        st.write(result)

        st.subheader("تحليل الموضة:")
        st.write(fashion_analysis(result))