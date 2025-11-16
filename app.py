import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
model = tf.keras.models.load_model("trained_plant_disease_model.keras")

# Function to predict
def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.resize((128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME","DISEASE RECOGNITION"])

# Display header image
img = Image.open("Diseases.png")
st.image(img, use_column_width=True)

# Home Page
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True
    )

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:", type=["png","jpg","jpeg"])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)
            
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f"Model predicts it is: {class_name[result_index]}")
