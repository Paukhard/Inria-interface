from google.cloud import storage
from google.oauth2 import service_account
import streamlit as st
import tensorflow as tf
import numpy as np

PROJECT_ID = 'le-wagon-bootcamp-398616'
LOCAL_API_DATA_FOLDER = ""
MAPS_API_KEY = st.secrets["MAPS_API_KEY"]
CREDENTIALS = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account_bastian"])

dim_dict = {
    "unet":(200,200,3),
    "segnet":(200,200,3),
    "ternaus":(200,200,3),
    "unet_complex":(512,512,3),
    "DeepLabV3":(512,512,3)
}

def loss03(y_true, y_pred):
    pass

custom_loss = {
    "unet":None,
    "segnet":None,
    "ternaus":loss03,
    "unet_complex":loss03,
    "DeepLabV3":None
}

@st.cache_resource
def get_model_from_gcs(model="unet"):

    print("Getting new model!")

    client = storage.Client(project=PROJECT_ID, credentials=CREDENTIALS)

    bucket_name = 'aerial_images_inria1358'

    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(f"h5 models/{model}.h5")

    blob.download_to_filename("model")

    if custom_loss[model] is not None:
        st.write("trying to load custom model")
        model = tf.keras.models.load_model("model", custom_objects={'loss': loss03})
    else:
        st.write("NOT!! trying to load custom model")
        model = tf.keras.models.load_model("model")


    return model

def compute_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) of two binary segmentation masks.

    Args:
        mask1 (numpy.ndarray): First binary mask.
        mask2 (numpy.ndarray): Second binary mask.

    Returns:
        float: IoU score.
    """
    # Ensure the masks are binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Intersection and Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union
