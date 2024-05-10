'''
This is the main script to run the Streamlit app to display the evaluation results MOT16 train dataset.
'''
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
import base64
from pyecharts.charts import Bar
from pyecharts import options as opts
from werkzeug.utils import secure_filename
from multiprocessing import freeze_support
from evaluate_filtered_frames import run_evaluation
from image_generator import generate_image, image_check, generate_image_zoomable
from charts import create_bar_chart
import os

# ===========================Streamlit Setup============================
# Set page title and layout
st.set_page_config(layout="wide", page_title="Vis Your MoT Model!")
st.write("##Visualize your model performance")

# Set max file size and acceptable file types
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB=

# Display the sidebar
st.sidebar.write("## Select the video sequence you want to evaluate and upload your model outcome!")

if 'page' not in st.session_state:
    st.session_state.page = 1

# ==============Select Video, Upload Model Outcome, and Select Frame to Evaluate =====================
col1, col2 = st.columns(2)

# Video selection
dict_video_sequence = {'MOT16-02': 600, 'MOT16-04': 1050, 'MOT16-05': 837, 'MOT16-09': 525, 'MOT16-10': 654, 'MOT16-11': 900, 'MOT16-13': 750}
video_sequence = st.sidebar.selectbox(
    'Video Sequence You Want To Explore',
    (f'MOT16-02', f'MOT16-04', f'MOT16-05', f'MOT16-09', f'MOT16-10', f'MOT16-11', f'MOT16-13'),
    index=0,
)

# Upload the model outcome
my_upload = st.sidebar.file_uploader("Upload your model outcome", type=["txt"])
# save the file to the local directory
if my_upload is not None:
    # To read file as string:
    file_details = {"FileName": video_sequence, "FileType": "txt", "FileSize": my_upload.size}
    # Save the file to disk
    with open((my_upload.name), "wb") as f:
        f.write(my_upload.getbuffer())
    st.success(f"the txt file uploaded for video sequence {video_sequence} has been saved")

# Frame selection
n_frames = dict_video_sequence[video_sequence]

# Frame selection using session state to trigger re-evaluation
if 'frame_range' not in st.session_state:
    st.session_state.frame_range = (1, dict_video_sequence[video_sequence])

new_range = st.slider('Select a range of values', 1, dict_video_sequence[video_sequence], st.session_state.frame_range, key='values')
st.write(f"Selected frame range: {new_range}")
if new_range != st.session_state.frame_range:
    st.session_state.frame_range = new_range
    st.session_state.evaluate = True  # Trigger evaluation

# ===========================Image Check and Display============================
# Image check and display
if image_check(my_upload, MAX_FILE_SIZE):
    generate_image_zoomable(my_upload=my_upload, video_sequence = video_sequence, values = new_range, col1=col1, col2=col2)

# Perform evaluation if the range has changed
if 'evaluate' in st.session_state and st.session_state.evaluate:
    t0, t1 = st.session_state.frame_range
    st.session_state.evaluate = False  # Reset the flag
    with st.spinner('Evaluation ongoing for the selected frame range...'):
        result = run_evaluation(t0=t0, t1=t1, SEQ_INFO=video_sequence, uploaded_txt_dir=my_upload.name)

        # Render and display charts
        tracking_chart = create_bar_chart(result['HOTA'], "Tracking Quality (HOTA)", "#5470C6")
        detection_chart = create_bar_chart(result['CLEAR'], "Detection Quality (CLEAR)", "#91CC75")
        identification_chart = create_bar_chart(result['Identity'], "Identification Quality (Identity)", "#EE6666")
        VACE_chart = create_bar_chart(result['VACE'], "VACE", "#fac858")
        count_chart = create_bar_chart(result['COUNT'], "Count", "#73c0de")

        components.html(tracking_chart, height=600)
        components.html(detection_chart, height=600)
        components.html(identification_chart, height=600)
        components.html(VACE_chart, height=600)
        components.html(count_chart, height=600)
        
        st.success("Evaluation done! You can select new range to re-evaluate.")

if st.button('Clear Evaluation'):
    # delete the saved file
    if my_upload is not None:
        os.remove(my_upload.name)
        # delete every image in the GT_plot and your_model_plot folder except the .gitignore file
        for file in os.listdir('GT_plot'):
            if file != '.gitignore':
                os.remove(f'GT_plot/{file}')
        for file in os.listdir('your_model_plot'):
            if file != '.gitignore':
                os.remove(f'your_model_plot/{file}')
        st.success(f'the txt file uploaded for video sequence {video_sequence} has been deleted')























