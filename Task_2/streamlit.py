import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fine_tuned.h5")  
    return model

model = load_model()

# Class names (Ensure they match your dataset)
class_names = ['Cas', 'Cos', 'Gum', 'MC', 'OC', 'OLP', 'OT']  
st.title("Teeth Classification AI")
st.write("Upload an image to classify the type of teeth.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Normalize for ResNet50
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100



    st.write(f"### Prediction: {class_names[predicted_class]}")
    st.write(f"### Confidence: {confidence:.2f}%")
