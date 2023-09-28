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

from utils import get_model_from_gcs, compute_iou, dim_dict, PROJECT_ID, CREDENTIALS
from maptiler import get_image_in_right_dimensions


st.set_page_config(
    layout="wide",
    page_title="Building Predictor",
    page_icon="🏛️",)


# TOP BAR
set = st.sidebar.selectbox('What set do you want to use?', ('train', 'test'))
type = st.sidebar.selectbox('Individual patch, or whole image?', ('whole_image', 'patch'))
filename = st.sidebar.text_input("File", "austin1.tif")
threshold = st.sidebar.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.5)
model_selection = st.sidebar.selectbox('What model do you want to use?', ('unet', 'segnet', 'DeepLabV3'))

show_iou = st.sidebar.checkbox('Show IOU graph')

# The predict button comes after the definition of the predict function

# PREDICT FUNCTION
def prediction():

    # SET HEADER
    st.header(f"Prediction for {filename}")

    columns = st.columns(3)

    dimensions = dim_dict[model_selection]

    # LOADING MODEl
    with st.spinner("Loading model..."):
        model = get_model_from_gcs(model_selection)

    with st.spinner("Getting input image from Google Cloud..."):
        if type == "patch":
            original = get_single_patch_im_array_from_gcloud(filename=filename, set=set, subset="images")
        else:
            original = get_im_array_from_gcloud(filename=filename, set=set, subset="images")

    with st.spinner("Making prediction"):
        prediction = get_prediction_image(original, model, dimensions=dimensions)

    with st.spinner("Getting ground truth from Google Cloud..."):
        if set == "train":
            if type == "patch":
                gt = get_single_patch_im_array_from_gcloud(filename=filename, set=set, subset="gt")
            else:
                gt = get_im_array_from_gcloud(filename=filename, set=set, subset="gt")

    with st.spinner('Showing prediction...'):
        # SHOW PREDICT MASK
        columns[2].write("Segmentation Mask")
        columns[2].image(((prediction)>threshold)*255)
        sub_columns = columns[2].columns(3)

        sub_columns[1].metric(label="Max value", value=f"{np.round(np.max(prediction)/255,3)}")
        sub_columns[2].metric(label="Min value", value=f"{np.round(np.min(prediction)/255,3)}")

        # IF ZOOM >= 17, THEN WE HAVE A GROUND TRUTH, SHOW GROUND TRUTH
        if set == "train":

            columns[0].write("Input Image")
            columns[0].image(original)

            columns[1].write("Ground Truth")
            columns[1].image(gt)

            # CALCULATE CHANGE TO PREVIOUS METRIC
            delta = None
            current_iou = compute_iou(prediction>threshold, gt)

            if 'prev_iou' in st.session_state:
                delta = f"{np.round((current_iou/st.session_state['prev_iou']-1)*100)}%"
                st.session_state['prev_iou'] = current_iou
            else:
                st.session_state['prev_iou'] = current_iou

            columns[1].metric(label="IOU", value=f"{current_iou}", delta=delta)
        else:
            columns[0].write("Input Image")
            columns[0].image(original)
        # IF WE WANT TO SHOW IOU GRAPH
        if show_iou:
            fig, ax = plt.subplots()

            linspace = np.linspace(0.05, 0.95, num=50)
            y = [compute_iou(prediction>ts, gt) for ts in linspace]
            ax.plot(linspace, y)
            ax.set_ylabel("IOU")
            ax.set_xlabel("Threshold")

            columns[0].pyplot(fig)



# SIDEBAR BUTTON
st.sidebar.button("Predict Buildings", on_click=prediction)

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

def get_im_array_from_gcloud(filename, set="train", subset="images"):

    storage_client = storage.Client(project=PROJECT_ID, credentials=CREDENTIALS)

    # Get the bucket
    bucket = storage_client.get_bucket("aerial_images_inria1358")

    blob = bucket.blob(f"AerialImageDataset/{set}/{subset}/{filename}")

    blob.download_to_filename(filename)

    return np.asarray(Image.open(filename))

def get_single_patch_im_array_from_gcloud(filename, set="train", subset="images", dimensions=(200,200,3)):

    patch_path = "Patches2" if dimensions==(200,200,3) else "Patches500"

    storage_client = storage.Client(project=PROJECT_ID, credentials=CREDENTIALS)

    # Get the bucket
    bucket = storage_client.get_bucket("aerial_images_inria1358")

    blob = bucket.blob(f"{patch_path}/{set}/{subset}/{filename}")

    blob.download_to_filename(filename)

    return np.asarray(Image.open(filename))
