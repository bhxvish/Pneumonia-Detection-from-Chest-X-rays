import streamlit as st
from predict import predict
st.title("Pneumonia Detector from X-rays")
file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if file:
    with open("temp.jpg","wb") as f:
        f.write(file.getbuffer())
    st.image("temp.jpg", caption="Upload Image", use_container_width=True)
    result = predict("temp.jpg")
    st.success(f"Prediction: {result}")