import os
# os.environ['SM_FRAMEWORK'] = 'tf.keras'
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model
# from keras import backend as K
import numpy as np
import tifffile
# import segmentation_models as sm
# from tensorflow.keras import utils
import matplotlib.pyplot as plt
from matplotlib import cm
import utils_v2
from model_utils import *

@st.cache_resource
def model_load(model_path):
    model = load_model(model_path,compile=False)
    return model


def prediction(model,image,class_names):
    # img_array = tf.keras.preprocessing.image.img_to_array(image)
    #st.write(img_array)
    # img_array = np.divide(img_array - img_array.min(), img_array.max() - img_array.min())
    #st.write(img_array)
    # img_array = tf.expand_dims(img_array, 0)
    
    #############
    image_in = load_image(image)
    img_array = preprocess_image(image_in)
    #############
    predicted_probs = model.predict(img_array)
    predicted_mask = np.argmax(predicted_probs, axis=-1)
    #st.write(predicted_mask)
    return predicted_mask, predicted_probs

def calculate_class_confidence(y_pred):
    confidence_scores = []
    num_classes = y_pred.shape[-1]  
    confidence_scores = np.mean(y_pred, axis=(1, 2))  # Average probability per class across all pixels
    return confidence_scores

def display_class_confidence(y_pred, class_names):
    num_classes = len(class_names)
    confidence_scores = np.mean(y_pred, axis=(1, 2))  # Average probability per class across all pixels

    st.write("**Proportions of each class:**")
    for class_idx, class_name in enumerate(class_names):
        confidence_score = confidence_scores[0,class_idx] * 100
        confidence_score = "{:.2f}%".format(confidence_score)
        st.info(f"Proportion of {class_name}: {confidence_score}")

def main_predict():
    st.title("Detecting the seagrass presence")
    st.markdown("This app is for prediction of seagrass in the mediterranean sea.")
    chosen_region = st.sidebar.selectbox("Choose the region",['','Greece','Croatia'])
    image_file = st.file_uploader("Drop the picture of the location",type = ['tif'])
    st.markdown("Choose the region of the picture you are uploading in the sidebar.")

    class_names = ['seagrass','water','land']
    if chosen_region == "Greece":
        # model = model_load('./saved_models/unet_cleaned_summer_V1.h5')
        model = utils_v2.retrieve_model()
    elif chosen_region == "Croatia":
        model = model_load('./saved_models/unet_wcc_summer_croatia.h5')

    if chosen_region:
        if image_file is not None:
            with open("temp.tif", "wb") as f:
                f.write(image_file.read())

            # Read the TIFF image from the saved file
            col1, col2 = st.columns(2)
            image = tifffile.imread("temp.tif")
            # shape = image.shape
            # st.write("shape of the image is: " , shape)
            # Ensure that pixel values are within the range [0, 255]
            r_image = np.clip(image, 0, 255)
            # If image has more than 4 channels, take the first 3 channels
            if image.shape[-1] > 4:
                rgb_image = r_image
                img_array = np.divide(rgb_image - rgb_image.min(), rgb_image.max() - rgb_image.min())
                #rgb_image_uint8 = (rgb_image / np.max(rgb_image) * 255).astype(np.uint8)
                # st.write("RGB image shape:", rgb_image.shape) 
                # st.write("Data type:", rgb_image_uint8.dtype)
            else:
                rgb_image = r_image

            with col1:
                st.subheader('Original Image')
                st.image(img_array[:,:,[3,2,1]], caption="Uploaded Image", use_column_width=True)

            st.sidebar.markdown("Click on the predict button below to predict the mask of the image.")
            button = st.sidebar.button("Predict")

            if button:

                predicted_mask, predicted_probs = prediction(model, image, class_names)
                predicted_mask_2d = np.squeeze(predicted_mask, axis=0)
                cmap = plt.cm.get_cmap('viridis', 3) 
                cmap = cm.get_cmap(cmap)
                rgba_image = cmap(predicted_mask_2d)

                # Display masked image in Streamlit
                with col2:
                    st.subheader('Masked Image')
                    st.image(rgba_image, caption='Masked Image', use_column_width=True)

                st.write("The purple color in the mask is the seagrass, blue color is the water and yellow color is the land.")
                display_class_confidence(predicted_probs, class_names)
                # st.write(predicted_probs)
                # st.write(predicted_probs.shape)
                
            os.remove("temp.tif")
        
    else:
        st.write("Select the region")

# if __name__=="__main__":
    # main()

