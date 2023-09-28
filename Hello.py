import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to the Inria-1358 Project 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    Our final project for the Le Wagon Bootcamp was about classifying buildings on Satellite Images.
    We have prepared 3 use-cases
    - Using images from the original dataset
    - Using single images from the Google Maps API (password-protected)
    - Using tiled images from the Google Maps API (password-protected)
    **👈 Select a page from the sidebar** to explore our project
    ### Want to learn more?
    - Check out [the inria challenge](https://streamlit.io)
    - Jump into our [code](https://github.com/bergerbastian/inria1358)
"""
)
