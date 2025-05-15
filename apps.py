import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import subprocess
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ---- Load the trained model ----
MODEL_PATH = r"C:\Users\Nethra\Downloads\BLNW\BLNW\blnw_cnn_model.h5"  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (must match dataset)
class_names = ['bolt', 'locatingpin', 'nut', 'washer']

# ---- Streamlit UI ----
st.title("Mechanical Parts Detector 🔧")
st.sidebar.title("Upload Image or Use Webcam")
choice = st.sidebar.radio("Select Input Mode", ("Manual Mode", "Automatic Mode"))

def predict_image(image):
    """Preprocess the image and get predictions."""
    img_array = img_to_array(image)
    img_array = cv2.resize(img_array, (224, 224))  # Ensure correct input size
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    return predicted_class, confidence

def estimate_dimensions(image, predicted_class):
    """Estimate specific dimensions based on component type."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        object_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(object_contour)

        # Define known reference width for each class (in mm)
        reference_sizes = {
            "bolt": 13,         # Bolt head width
            "nut": 13,          # Nut across flats
            "washer": 12,       # Washer outer diameter
            "locatingpin": 10   # Pin diameter
        }

        if predicted_class in reference_sizes:
            known_size_mm = reference_sizes[predicted_class]
            scale = known_size_mm / w  # mm per pixel

            real_w = w * scale
            real_h = h * scale

         # Component-specific logic
        if predicted_class == "bolt":
            return (
        f"Bolt Head Width: {real_w:.2f} mm\n"
        f"Shaft Length: {real_h:.2f} mm"
    )

        elif predicted_class == "nut":
            return (
        f"Width Across Flats: {real_w:.2f} mm\n"
        f"Thickness: {real_h:.2f} mm"
    )
        elif predicted_class == "locatingpin":
            return (
        f"Pin Diameter: {real_w:.2f} mm\n"
        f"Length: {real_h:.2f} mm"
    )

        elif predicted_class == "washer":
    # Estimate inner diameter using minimum enclosing circle (assumes round shape)
            (cx, cy), radius = cv2.minEnclosingCircle(object_contour)
    outer_d = 2 * radius * scale
    inner_d = outer_d * 0.5  # Approximate (or calibrate with sample washer)
    return (
        f"Outer Diameter: {outer_d:.2f} mm\n"
        f"Inner Diameter: {inner_d:.2f} mm"
    )

# ---- Image Upload Mode ----
if choice == "Manual Mode":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))  # Ensure resizing
        st.image(image, caption="Uploaded Image", width=200)  # Smaller image
        
        predicted_class, confidence = predict_image(image)
        st.markdown(f'<p style="font-size:30px; color:green;"><strong>Detected: {predicted_class} ({confidence:.2f}%)</strong></p>', unsafe_allow_html=True)

        # Convert PIL image to OpenCV format for processing
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        
        # Estimate dimensions
        dimensions = estimate_dimensions(image_cv, predicted_class)
        formatted_dimensions = dimensions.replace("\n", "<br>")
        st.markdown(f'<p style="font-size:25px; color:blue;"><strong>{formatted_dimensions}</strong></p>', unsafe_allow_html=True)

        #st.markdown(f'<p style="font-size:25px; color:blue;"><strong>{dimensions}</strong></p>', unsafe_allow_html=True)

# ---- Webcam Mode ----
def run_detection_script():
    script_path = r"C:\Users\Nethra\Downloads\finial nut\finial nut\detection.py"
    if os.path.exists(script_path):
        subprocess.run(f'code -r {script_path} && start cmd /k "cd {os.path.dirname(script_path)} && py detection.py"', shell=True)
    else:
        st.error("Error: detection.py not found at the specified path!")

if choice == "Automatic Mode":
    if st.button("Open Webcam"):
        run_detection_script()