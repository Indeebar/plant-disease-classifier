import streamlit as st
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")
import os
import requests
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np



# Check and download the model if not exists
model_path = "plant_disease_model.h5"
if not os.path.exists(model_path):
    # Google Drive direct download via gdown
    import gdown
    file_id = "1-3nE1XiZLK--iinHtT0H1b5paZZLecsO"
    url = f"https://drive.google.com/uc?id={file_id}"
    st.info("⬇️ Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

# Class labels (38)
class_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Preprocessing
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Prediction
def predict_image_class(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]
    return class_labels[predicted_index], confidence, prediction

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")

with st.sidebar:
    st.title("🌾 PlantDoc")
    st.markdown("Upload an image of a plant leaf to detect possible diseases using a CNN model.")
    st.markdown("📷 Supported: `.jpg`, `.jpeg`, `.png`")
    st.markdown("---")
    st.markdown("🤖 **Model:** Custom CNN (38 classes)")
    st.markdown("🔬 **Input size:** 128x128")
    st.markdown("📈 **Output:** Disease prediction and confidence")

st.title("🌿 Plant Disease Detection App")
st.caption("Identify plant diseases from leaf images in seconds.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Leaf Image", use_column_width=True)

    if st.button("🔍 Diagnose"):
        with st.spinner("Analyzing the image..."):
            label, confidence, prediction_array = predict_image_class(image)

        st.success("✅ Prediction Complete!")
        st.markdown(f"**🧪 Detected Disease:** `{label}`")
        st.markdown(f"**📊 Confidence:** `{confidence * 100:.2f}%`")

        top_k = 5
        top_indices = np.argsort(prediction_array)[-top_k:][::-1]
        top_labels = [class_labels[i] for i in top_indices]
        top_scores = [prediction_array[i] for i in top_indices]

        st.markdown("### 🔝 Top 5 Class Probabilities")
        chart_data = {
            "Disease": top_labels,
            "Confidence": [round(score * 100, 2) for score in top_scores]
        }
        st.bar_chart(chart_data, x="Disease", y="Confidence")

else:
    st.info("👈 Upload an image from the sidebar to get started.")


