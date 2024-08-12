import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# Load the trained model
model_path = 'C:/Users/zoezh/Downloads/mscnn_fabric_defect_detector.keras'
model = tf.keras.models.load_model(model_path)

def preprocess_image_opencv(image):
    image_np = np.array(image)

    # Resize image using OpenCV
    image_resized = cv2.resize(image_np, (128, 128))

    # Convert to grayscale
    if len(image_resized.shape) == 3:  # Check if the image has color channels
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_resized

    # Normalize the image
    image_gray = image_gray / 255.0

    # Add channel dimension for model compatibility
    image_gray = np.expand_dims(image_gray, axis=-1)
    image_gray = np.expand_dims(image_gray, axis=0)  # Add batch dimension
    return image_gray

# Function to predict if the image contains defects
def predict_defect(image_gray):
    # Predict using the model
    prediction = model.predict(image_gray)
    
    # Convert prediction to binary output (0 or 1)
    return "Defect" if prediction[0] > 0.5 else "No Defect"

# Streamlit app
st.title('Fabric Defect Detection')

st.write("Upload an image of fabric to detect defects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image using PIL
    image = Image.open(uploaded_file)
    
    # Preprocess the image
    preprocessed_image = preprocess_image_opencv(image)
    
    # Predict defect
    result = predict_defect(preprocessed_image)
    
    # Display the uploaded and preprocessed image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f'Prediction: {result}')
