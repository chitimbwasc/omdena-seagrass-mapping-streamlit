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
import cv2
from scipy import stats

@st.cache_resource
def model_load(model_path):
    model = load_model(model_path,compile=False)
    return model

# Helper preprocessing functions

# Helps remove sun glint from images
def deglint_image(image_arr, mask_arr):
    deglinted_img = np.zeros(image_arr.shape)
    for i in range(7):
        band_i = image_arr[:, :, i]
        nir_band = image_arr[:, :, 7]
        slope, y_inter, r_val, p_val, std_err = stats.linregress(x=nir_band.ravel(), y=band_i.ravel())
    
        water_mask = ((mask_arr==0) | (mask_arr==1))
        deglinted_img[:, :, i][water_mask] = band_i[water_mask] - (slope*(nir_band[water_mask] - nir_band[water_mask].min()))
        deglinted_img[:, :, i] = cv2.medianBlur(deglinted_img[:, :, i].astype('float32'), 3)
        deglinted_img[:, :, i][~water_mask] = band_i[~water_mask]
        deglinted_img[:, :, i] = np.where(deglinted_img[:, :, i] < 0, 0, deglinted_img[:, :, i])
    deglinted_img[:, :, 7:] = image_arr[:, :, 7:]
    return deglinted_img

# Helps calculate Depth invariant index between 2 bands using Lyzenga method 
def calculate_dii(band_1, band_2, mask_arr):
    band_1_transformed = np.log(band_1 + 1)
    band_2_transformed = np.log(band_2 + 1)
    water_mask = ((mask_arr==0) | (mask_arr==1))
    dii = np.zeros(band_1.shape)
    
    cov_matrix = np.cov(band_1_transformed[water_mask].ravel(), band_2_transformed[water_mask].ravel())
    a = (cov_matrix[0, 0] - cov_matrix[1, 1]) / (cov_matrix[0, 1])
    att_coef_ratio = a + np.sqrt(a**2 + 1)
    
    dii[water_mask] = band_1_transformed[water_mask] - (att_coef_ratio * band_2_transformed[water_mask])
    dii[~water_mask] = (band_1_transformed[~water_mask] + band_2_transformed[~water_mask])/2
    return dii

def swm_land_mask(img_arr, threshold):
    # start = time.time()
    swm = (img_arr[:, :, 1] + img_arr[:, :, 2]) / (img_arr[:, :, 7] + img_arr[:, :, 11] + 1e-3)
    mask = swm < threshold
    mask_arr = cv2.medianBlur(mask.astype('float32'), 3)
    # end = time.time()
    # print(f'NDVI took: {end-start} secs')
    return mask_arr

def preprocess_image_mask(image_arr, mask_arr):
    # Deglint the image (bands 1-6 using band 8 as reference)
    deglinted_image = deglint_image(image_arr, mask_arr)

    # Apply water column correction 
    dii_b2_b3 = calculate_dii(deglinted_image[:, :, 1], deglinted_image[:, :, 2], mask_arr)
    image_arr_final = np.dstack((deglinted_image, dii_b2_b3))

    # Image preprocessing
    min_vals = image_arr_final.min(axis=(0, 1), keepdims=True) # image_arr shape is (256, 256, num_bands)
    max_vals = image_arr_final.max(axis=(0, 1), keepdims=True)
    image_arr_normalized = (image_arr_final - min_vals) / (max_vals - min_vals + 1e-3)
    
    # Mask preprocessing
    mask_arr_processed = tf.one_hot(mask_arr, depth=3).numpy()

    return image_arr_normalized.astype('float32'), mask_arr_processed.astype('float32')

def set_shapes(img, mask):
    img.set_shape([256, 256, 13])
    mask.set_shape([256, 256, 3])
    return img, mask

def prediction(model,image,class_names):
    # img_array = tf.keras.preprocessing.image.img_to_array(image)
    # #st.write(img_array)
    # img_array = np.divide(img_array - img_array.min(), img_array.max() - img_array.min())
    # #st.write(img_array)
    #img_array = tf.expand_dims(image, 0)

    predicted_probs = model.predict(image)
    predicted_mask = np.argmax(predicted_probs, axis=-1)
    #st.write(predicted_mask)
    return predicted_mask, predicted_probs

def calculate_class_confidence(y_pred):
    confidence_scores = []
    num_classes = y_pred.shape[-1]  

    # for class_idx in range(num_classes):
    #     class_pixels = np.sum(y_pred[..., class_idx] == class_idx)  
    #     total_pixels = np.prod(y_pred.shape)
    #     confidence_score = class_pixels / total_pixels
    #     confidence_scores.append(confidence_score)

    class_probs = np.mean(y_pred, axis=(1, 2))  # Average probability per class across all pixels
    confidence_scores = class_probs
    return confidence_scores

def display_class_confidence(y_pred, class_names):
    num_classes = len(class_names)
    confidence_scores = calculate_class_confidence(y_pred)

    st.write("**Proportions of each class:**")
    for class_idx, class_name in enumerate(class_names):
        confidence_score = confidence_scores[0,class_idx] * 100
        confidence_score = "{:.2f}%".format(confidence_score)
        st.info(f"Proportion of {class_name}: {confidence_score}")

def main():
    st.title("Detecting the seagrass presence")
    st.markdown("This app is for prediction of seagrass in the mediterranean sea.")
    chosen_region = st.sidebar.selectbox("Choose the region",['','Greece','Croatia'])
    image_file = st.file_uploader("Drop the picture of the location",type = ['tif'])
    st.markdown("Choose the region of the picture you are uploading in the sidebar.")

    class_names = ['seagrass','water','land']
    if chosen_region == "Greece":
        model = model_load('./saved_models/unet_cleaned_summer_V1.h5')
    elif chosen_region == "Croatia":
        model = model_load('./saved_models/unet_wcc_summer_croatia.h5')

    if chosen_region:
        if image_file is not None:
            with open("temp.tif", "wb") as f:
                f.write(image_file.read())

            # Read the TIFF image from the saved file
            col1, col2 = st.columns(2)
            image = tifffile.imread("temp.tif")
            shape = image.shape
            #st.write("shape of the image is: " , shape)
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
                st.image(img_array[:,:,2], caption="Uploaded Image", use_column_width=True)

            st.sidebar.markdown("Click on the predict button below to predict the mask of the image.")
            button = st.sidebar.button("Predict")

            if button:
                st.write("Original shape",image.shape)
                mask_arr = swm_land_mask(image, threshold=1.6)
                image, mask = preprocess_image_mask(image, mask_arr)
                st.write("After preprocessing",image.shape)
                #image, mask  = set_shapes(image, mask)
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

if __name__=="__main__":
    main()


