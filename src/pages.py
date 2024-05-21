import streamlit as st
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
from classify import main

# Utility functions
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

# Styling function
def make_font_poppins():
    with open("src/styles.css") as css:
        st.markdown(f'<style>{css.read()}</style>' , unsafe_allow_html= True)

    # Render the custom CSS style
    st.markdown("<link href='https://fonts.googleapis.com/css2?family=Poppins:wght@500&display=swap' rel='stylesheet'>", unsafe_allow_html=True)

    # Hide Streamlit style
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Image display function
def display_image(url, caption=None):
    st.markdown("<p></p>",unsafe_allow_html=True)
    image_url = url
    st.image(image_url, caption=caption, use_column_width=True)
    st.markdown("<p></p>",unsafe_allow_html=True)

# Home page display function
def homepage():
    ################### HEADER SECTION #######################
    display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')
    st.markdown("<h1 style='text-align: center; color: auto;'>Mapping Seagrass Meadows with Satellite Imagery and Computer Vision</h1>",
                unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: auto; font-family: Segoe UI;'>A web-based pixel-level Classification Model for identifying seagrass in sattelite images</h3>", unsafe_allow_html=True)
    display_image('https://upload.wikimedia.org/wikipedia/commons/4/45/Sanc0209_-_Flickr_-_NOAA_Photo_Library.jpg')

# Information display function
def about():
    ################### HEADER SECTION #######################
    display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')
    
    ################### INFORMATION SECTION #######################
    st.header('Challenge Background')
    st.markdown(
                """
                <style>
                .tab {
                    text-indent: 0px;  /* adjust as needed */
                    text-align: justify;  /* Add this line */
                }
                </style>
                <div class="tab" style="text-align=justify;">Seagrasses, forming vast marine meadows in shallow salt waters from tropics to the Arctic, are vital for biodiversity. They provide habitats for fish and shellfish, supporting local coastal economies. Moreover, they stabilise sediment, absorb wave energy, and contribute significantly to carbon absorption, making them crucial allies in combating climate change.</div>
                <p></p>
                
                """
                ,unsafe_allow_html=True)
    
    st.header('The Problem')
    st.markdown(
                """
                <style>
                .tab {
                    text-indent: 0px;  /* adjust as needed */
                    text-align: justify;  /* Add this line */
                }
                </style>
                <div class="tab" style="text-align=justify;">The declining health of Posidonia oceanica meadows in the Mediterranean Sea is a pressing concern, attributed to climate change and various human activities. These meadows, crucial for their ecological significance, are in jeopardy due to factors such as warming, ocean acidification, coastal urban development, fishing, and aquaculture. This decline has led to a substantial loss of goods and services provided by these ecosystems. While P. oceanica is the most vital and studied seagrass species in the region, there has been a limited effort to collate and provide a comprehensive distribution of these meadows. This lack of information impedes our ability to effectively address the regression of these critical habitats.</div>
                <p></p>
                
                """
                ,unsafe_allow_html=True)
    
    st.header('Goal of the Project')
    st.markdown(
                """
                <style>
                .tab {
                    text-indent: 0px;  /* adjust as needed */
                    text-align: justify;  /* Add this line */
                }
                </style>
                <div class="tab" style="text-align=justify;">Our project aims to develop accessible and efficient methods for mapping seagrass meadows using readily available satellite imagery and  computer vision. We envision these results as valuable tools for long-term seagrass monitoring, including tracking restoration and replanting efforts. Our primary goal is to create a pixel-level classification and segmentation model to map seagrass distribution, with a focus on the Mediterranean, especially Italian waters. Using computer vision techniques, we'll identify seagrass regions by analyzing satellite images and classifying pixels based on data from public databases indicating seagrass presence or absence. An essential aspect of this project involves comparing our model's outcomes with established habitat suitability models for P. oceanica presence. Habitat suitability models predict species presence in a location by analyzing the relationship between observed occurrences and environmental conditions. They assess marine habitat status, forecast species distribution changes from human and environmental impacts, and guide restoration efforts by identifying optimal areas.</div>
                <p></p>
                
                """
                ,unsafe_allow_html=True)
    
    make_font_poppins()

# Classification function
def classify():
    ################### HEADER SECTION #######################
    display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')
    
    img_file = st.file_uploader("Choose an image to classify", type=["tif"])
    main(img_file)

# Placeholder function
def guide():
    ################### HEADER SECTION #######################
    display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')
    
    ################### INFORMATION SECTION #######################

# Placeholder function
def datasets():
     ################### HEADER SECTION #######################
    display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')
    
    ################### INFORMATION SECTION #######################

# Placeholder function
def repository():
    ################### HEADER SECTION #######################
    display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')
    
    ################### INFORMATION SECTION #######################

# Loading the and displaying images
def show_image(image_path, band_idxs=range(12)):
    # Load the image
    img_arr = tiff.imread(image_path)[:, :, band_idxs]
    
    # Display the image in streamlit
    fig, ax = plt.subplots()
    ax.imshow(img_arr[:, :, 3])
    st.pyplot(fig)
            
# Placeholder function
def gallery():
     ################### HEADER SECTION #######################
    display_image('https://cdn-images-1.medium.com/max/800/0*vBDO0wwrvAIS5e1D.png')
    
    ################### INFORMATION SECTION #######################
    import streamlit as st

    st.title("Gallery :film_frames:")
    st.markdown("*This page displays the images, masks and their predictions for Greece and Croatia regions*")

    tab1, tab2 = st.tabs(['Greece','Croatia'])

    with tab1:
        # st.image("./model_utils/Croatia_images/image1.JPG", caption='Croatia image1')
        # st.image("./model_utils/Croatia_images/image2.JPG", caption='Croatia image1')
        # st.image("./model_utils/Croatia_images/image3.JPG", caption='Croatia image1')
        # st.image("./model_utils/Croatia_images/image4.JPG", caption='Croatia image1')
        # st.image("./model_utils/Croatia_images/image5.JPG", caption='Croatia image1')
        
        # pass

        import os

        # Get the current working directory
        current_directory = os.getcwd()

        # Print the current working directory
        st.write("Current Working Directory:", current_directory)


    with tab2:
        # show_image("./src/images/Croatia_01_image_0.tif")
        # show_image("./src/images/Croatia_01_image_1.tif")
        # show_image("./src/images/Croatia_01_image_10.tif")
        # show_image("./src/images/Croatia_01_image_100.tif")
        # show_image("./src/images/Croatia_01_image_101.tif")

        # Get the current working directory
        current_directory = os.getcwd()
        folder_path = current_directory + "/src/images/"
        image_file_names = gallery_display(current_directory)
        for file_name in image_file_names:
            # show_image(current_directory + "/" + file_name)
            st.write(folder_path + file_name)


def gallery_display(dir_path):
            
    import os

    # Specify the directory path
    directory_path = dir_path

    # Create a list to store the file names
    file_list = []

    # Use os.scandir() to get the list of entries in the specified path
    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_file():
                file_list.append(entry.name)

    # Print the list of files
    # print("Files in directory:", file_list)

    return file_list
