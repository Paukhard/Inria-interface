from google.cloud import storage
from google.oauth2 import service_account
import streamlit as st
import tensorflow as tf

@st.cache_resource
def get_model_from_gcs(model="unet"):

    print("Getting new model!")

    credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
    )

    client = storage.Client(project="wagon-taxi-cab", credentials=credentials)
    buckets = client.list_buckets()

    bucket_name = 'taxifare_paukhard'
    directory_name = 'unet'
    destination_folder = 'unet'

    bucket = client.get_bucket(bucket_name)

    blob_iterator = bucket.list_blobs()

    blob = bucket.blob(f"models/{model}.h5")

    blob.download_to_filename("model")
    model = tf.keras.models.load_model("model")

    return model
