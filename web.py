import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model=tf.keras.models.load_image("trained_plant_disease_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size(128,128))
    imput_arr=tf.keras.preprocessing.image.img_to_array(image)
    imput_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("plant disease system for sustainabe agriculture")
app_mode=st.sidebar.selectbox('selectpage',['Home','disease recognition'])

from PIL import Image
img=image.open('Disease.png')
st.image(img)

if(app_mode=='HOME'):
    st.markdown("<h1 style='text-align: center;'>plant Disease Detection System for sustainable agriculture",unsafe_allow_html=True )