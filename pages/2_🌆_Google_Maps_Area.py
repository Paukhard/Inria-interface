import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import tensorflow as tf
from google.cloud import storage
from google.oauth2 import service_account

import tensorflow as tf
from stqdm import stqdm
from PIL import Image
import numpy as np
import urllib
import time

import os

from patchify import patchify

from utils import get_model_from_gcs, compute_iou, dim_dict, LOCAL_API_DATA_FOLDER, MAPS_API_KEY
import maptiler

st.set_page_config(
    layout="wide",
    page_title="Building Predictor",
    page_icon="🏛️",)


# TOP BAR
street = st.sidebar.text_input("Address", "Schützenstraße 40, Berlin")
zoom_level = st.sidebar.number_input("Zoom (can't be changed)", min_value=17, max_value=17, value=17, format="%i")
threshold = st.sidebar.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.5)
model_selection = st.sidebar.selectbox('What model do you want to use?', ('unet', 'segnet', 'DeepLabV3'))
password = st.sidebar.text_input("Password", type="password")

show_iou = st.sidebar.checkbox('Show IOU graph')

# The predict button comes after the definition of the predict function





# PREDICT FUNCTION
def prediction():
    if password != "123":
        st.write("Wrong password!")
        return

    # SET HEADER
    st.header(f"Prediction for {street}")

    # GET LOCATION DATA
    geolocator = Nominatim(user_agent="GTA Lookup")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street)

    lat = location.latitude
    lon = location.longitude

    columns = st.columns(3)

    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})

    dimensions = dim_dict[model_selection]

    area_id = "this_is_area_id"

    # LOADING MODEl
    with st.spinner("Loading model..."):
        model = get_model_from_gcs(model_selection)

    with st.spinner("Getting input image from Maps API..."):
        nrows, ncols = maptiler.get_tiling_images(area_id, center_Lat=lat, center_Lng=lon, zoom=zoom_level)
        original = maptiler.combine_tiling_images(area_id=area_id,n_rows=nrows, n_cols=ncols)
        original = maptiler.get_image_in_right_dimensions(original, dimensions)

    with st.spinner("Making prediction"):
        prediction = get_prediction_image(original, model, dimensions=dimensions)

    with st.spinner("Getting ground truth from Maps API..."):
        nrows, ncols = maptiler.get_tiling_images(area_id, center_Lat=lat, center_Lng=lon, ground_truth=True, zoom=zoom_level)
        gt = maptiler.combine_tiling_images(area_id=area_id,n_rows=nrows, n_cols=ncols, ground_truth=True)
        gt = maptiler.get_image_in_right_dimensions(gt, dimensions)

    with st.spinner('Showing prediction...'):
        # SHOW PREDICT MASK
        columns[2].write("Segmentation Mask")
        columns[2].image(((prediction)>threshold)*255)
        sub_columns = columns[2].columns(3)

        sub_columns[1].metric(label="Max value", value=f"{np.round(np.max(prediction)/255,3)}")
        sub_columns[2].metric(label="Min value", value=f"{np.round(np.min(prediction)/255,3)}")

        # IF ZOOM >= 17, THEN WE HAVE A GROUND TRUTH, SHOW GROUND TRUTH
        if zoom_level >= 17:

            columns[0].write("Input Image")
            columns[0].image(original)

            columns[1].write("Ground Truth")
            columns[1].image(gt)

            # CALCULATE CHANGE TO PREVIOUS METRIC
            delta = None
            current_iou = compute_iou(prediction>threshold, np.array(tf.squeeze(tf.image.rgb_to_grayscale(gt))))

            if 'prev_iou' in st.session_state:
                delta = f"{np.round((current_iou/st.session_state['prev_iou']-1)*100)}%"
                st.session_state['prev_iou'] = current_iou
            else:
                st.session_state['prev_iou'] = current_iou

            columns[1].metric(label="IOU", value=f"{current_iou}", delta=delta)

        # IF WE WANT TO SHOW IOU GRAPH
        if show_iou:
            fig, ax = plt.subplots()

            linspace = np.linspace(0.05, 0.95, num=50)
            y = [compute_iou(prediction>ts, np.array(tf.squeeze(tf.image.rgb_to_grayscale(gt)))) for ts in linspace]
            ax.plot(linspace, y)
            ax.set_ylabel("IOU")
            ax.set_xlabel("Threshold")

            columns[0].pyplot(fig)

    st.header("Location")
    st.map(map_data)


# SIDEBAR BUTTON
st.sidebar.button("Predict Buildings", on_click=prediction)


def get_input_image_maps(lat, lon, zoom=17, dimensions = (200,200, 3)):
    """Returns original image as a matrix from google maps.
    """
    image_url=f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&scale=2&maptype=satellite&key={MAPS_API_KEY}"

    image_path = LOCAL_API_DATA_FOLDER
    image_filename = f"{str(lat).replace('.','_')}__{str(lon).replace('.','_')}"
    image_type = "png"

    urllib.request.urlretrieve(image_url, f"input_{image_filename}.{image_type}")

    # Calculate max patches
    width = dimensions[0]
    height = dimensions[1]

    w_max_patches = int(1280 / width)
    h_max_patches = int(1280 / height)

    # Calculate max size
    w_max = width * w_max_patches
    h_max = height * h_max_patches

    # Required crop
    w_crop = 1280 - w_max
    h_crop = 1280 - h_max

    # Set crop boundaries
    left = w_crop / 2
    top = h_crop / 2
    right = 1280-w_crop/2
    bottom = 1280-h_crop/2

    # Open the downloaded image in PIL
    my_img = Image.open(f"input_{image_filename}.{image_type}").crop((left, top, right, bottom)).convert("RGB")

    os.remove(f"input_{image_filename}.{image_type}")

    imarray = np.array(my_img)

    return imarray

def get_prediction_image(imarray, model, dimensions = (200,200, 3)):
    patches = patchify(imarray, dimensions, step=dimensions[0])

    predict_data = []
    for r_ind in stqdm(range(patches.shape[0])):
        col_predict = []
        for c_ind in range(patches.shape[1]):
            image = patches[r_ind][c_ind]
            image = image/255

            # Predict
            predict_mask = model.predict(image, verbose=0)

            # Remove batch
            predict_mask = tf.squeeze(predict_mask)
            col_predict.append(predict_mask)
        predict_data.append(col_predict)

    rows = [np.hstack(predict_data[r]) for r in range(patches.shape[1])]
    return np.vstack(rows)



def predict_image_maps(lat, lon, model, zoom=17, return_ground_truth=True, dimensions = (200,200, 3)):
    """Returns original image and prediction matrix for a google maps image from the specified lat,lon. If zoom >= 17 and return_ground_truth = True, it also returns a GT generated from google maps api.
    """
    image_url=f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&scale=2&maptype=satellite&key={MAPS_API_KEY}"

    image_path = LOCAL_API_DATA_FOLDER
    image_filename = f"{str(lat).replace('.','_')}__{str(lon).replace('.','_')}"
    image_type = "png"

    urllib.request.urlretrieve(image_url, f"input_{image_filename}.{image_type}")

    # Calculate max patches
    width = dimensions[0]
    height = dimensions[1]

    w_max_patches = int(1280 / width)
    h_max_patches = int(1280 / height)

    # Calculate max size
    w_max = width * w_max_patches
    h_max = height * h_max_patches

    # Required crop
    w_crop = 1280 - w_max
    h_crop = 1280 - h_max

    # Set crop boundaries
    left = w_crop / 2
    top = h_crop / 2
    right = 1280-w_crop/2
    bottom = 1280-h_crop/2

    # Open the downloaded image in PIL
    my_img = Image.open(f"input_{image_filename}.{image_type}").crop((left, top, right, bottom)).convert("RGB")

    os.remove(f"input_{image_filename}.{image_type}")

    patch_list = []
    #im = Image.open(f'{image_path}')
    imarray = np.array(my_img)
    patches = patchify(imarray, dimensions, step=dimensions[0])

    predict_data = []
    for r_ind in stqdm(range(patches.shape[0])):
        col_predict = []
        for c_ind in range(patches.shape[1]):
            image = patches[r_ind][c_ind]
            image = image/255

            # Predict
            predict_mask = model.predict(image, verbose=0)

            # Remove batch
            predict_mask = tf.squeeze(predict_mask)
            col_predict.append(predict_mask)
        predict_data.append(col_predict)

    rows = [np.hstack(predict_data[r]) for r in range(patches.shape[1])]
    prediction = np.vstack(rows)



    if return_ground_truth and int(zoom) >= 17:
        return imarray, get_ground_truth(lat,lon, zoom, dimensions=dimensions), prediction
    else:
        return imarray, prediction


def get_ground_truth(lat, lon, zoom=17, dimensions = (200,200, 3)):
    gt_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size=640x640&scale=2&map_id=c2e4254a97b86e42&style=feature:all|element:labels|visibility:off&key={MAPS_API_KEY}"

    image_path = LOCAL_API_DATA_FOLDER
    image_filename = f"{str(lat).replace('.','_')}__{str(lon).replace('.','_')}"
    image_type = "png"
    urllib.request.urlretrieve(gt_url, f"gt_{image_filename}.{image_type}")

    # Calculate max patches
    width = dimensions[0]
    height = dimensions[1]

    w_max_patches = int(1280 / width)
    h_max_patches = int(1280 / height)

    # Calculate max size
    w_max = width * w_max_patches
    h_max = height * h_max_patches

    # Required crop
    w_crop = 1280 - w_max
    h_crop = 1280 - h_max

    # Set crop boundaries
    left = w_crop / 2
    top = h_crop / 2
    right = 1280-w_crop/2
    bottom = 1280-h_crop/2

    my_img = Image.open(f"gt_{image_filename}.{image_type}").crop((left, top, right, bottom)).convert("RGB")

    os.remove(f"gt_{image_filename}.{image_type}")


    # Load or create your image as a NumPy array
    image = np.array(my_img)  # Replace 'your_image' with your actual image array

    # Define the desired color and a tolerance
    desired_color = (191, 48, 191)  # Red in RGB
    tolerance = 80  # Adjust this tolerance as needed

    # Create a mask for the desired color
    lower_bound = np.array(desired_color) - tolerance
    upper_bound = np.array(desired_color) + tolerance
    mask = np.all((image >= lower_bound) & (image <= upper_bound), axis=2)

    # Color the masked pixels white
    result = np.zeros_like(image)
    result[mask] = (255,255,255)

    return result
