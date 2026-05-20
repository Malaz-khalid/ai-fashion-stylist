import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("🖤 AI Fashion Stylist")
st.write("ارفع صورة وسيتم تحليلها")

@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

model = load_model()

uploaded_file = st.file_uploader("ارفع صورة", type=["jpg","png","jpeg"])

def fashion_analysis(text):
    text = text.lower()
    score = 6

    if "white" in text:
        score += 1
    if "jeans" in text:
        score += 1

    return f"""
التقييم: {score}/10

الستايل: كاجوال بسيط

تحليل الإطلالة:
✔ إطلالة مناسبة

نصائح:
- إضافة إكسسوار بسيط
"""

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("تحليل"):
        result = model(image)[0]["generated_text"]

        st.subheader("الوصف:")
        st.write(result)

        st.subheader("تحليل الموضة:")
        st.write(fashion_analysis(result))