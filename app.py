import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from train import extract_features
import cv2

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    _, indices = neighbors.kneighbors([features])
    return indices

def show_recommend(image_path):
    features = extract_features(image_path)
    indices = recommend(features, feature_list)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.image(f'{filenames[indices[0][0]]}')
    with col2:
        st.image(f'{filenames[indices[0][1]]}')
    with col3:
        st.image(f'{filenames[indices[0][2]]}')
    with col4:
        st.image(f'{filenames[indices[0][3]]}')
    with col5:
        st.image(f'{filenames[indices[0][4]]}')
    with col6:
        st.image(f'{filenames[indices[0][5]]}')
    return True

st.title('Recommendation System')
st.subheader('Upload/capture an image to get recommendations')

if 'capture' not in st.session_state:
    st.session_state.capture = False
if 'start_video' not in st.session_state:
    st.session_state.start_video = False
if 'capture_image' not in st.session_state:
    st.session_state.capture_image = False
if 'captured' not in st.session_state:
    st.session_state.captured = False

capture = st.checkbox('Capture image', value=st.session_state.capture)

if capture:
    st.session_state.capture = True
    start_video = st.button('Start Video', on_click=lambda: setattr(st.session_state, 'start_video', True))
    capture_image = st.button('Capture Image', on_click=lambda: setattr(st.session_state, 'capture_image', True))
    st.markdown('---')
    stframe = st.empty()

    if st.session_state.start_video:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open video stream.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Could not read frame.")
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb)
                if st.session_state.capture_image:
                    cv2.imwrite('capture.jpg', frame)
                    st.session_state.captured = True
                    st.session_state.capture_image = False
                    st.session_state.start_video = False
                    cap.release()
                    cv2.destroyAllWindows()
                    break

    if st.session_state.captured:
        # st.image('capture.jpg')
        if show_recommend('capture.jpg'):
            st.success('Recommendation completed')
        st.session_state.captured = False

else:
    st.session_state.capture = False
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            display_image = Image.open(uploaded_file)
            st.image(display_image)
            if show_recommend(os.path.join("uploads", uploaded_file.name)):
                st.success('Recommendation completed')
        else:
            st.header("Some error occurred in file upload")
