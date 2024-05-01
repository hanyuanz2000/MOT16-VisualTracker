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
from image_generator import generate_image, image_check
from charts import create_bar_chart
import os

# ===========================Streamlit Setup============================
# Set page title and layout
st.set_page_config(layout="wide", page_title="Vis Your MoT Model!")
st.write("## 👀Visualize your model performance")

# Set max file size and acceptable file types
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

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
values = st.slider(
    'Select a range of values',
    1, n_frames, (1, n_frames), key='values'
)

# Show the range of frames selected
st.write(f"Start frame: {values[0]}", f", End frame: {values[1]}")

# ===========================Image Check and Display============================
# Image check and display
if image_check(my_upload, MAX_FILE_SIZE):
    generate_image(my_upload=my_upload, video_sequence = video_sequence, values = values, col1=col1, col2=col2)

# ===========================Evaluation============================
# Button to start evaluation
t0 = values[0]
t1 = values[1]

if st.button('Start Evaluation'):
    with st.spinner('Evaluation ongoing for the selected frame range...'):
        # Directly calling the evaluation function
        result = run_evaluation(t0 = t0, t1 = t1, SEQ_INFO = video_sequence, uploaded_txt_dir=my_upload.name)

        ### result have the following keys: ['HOTA', 'CLEAR', 'Identity', 'VACE', 'COUNT']
        tracking_quality = result['HOTA']
        detection_quality = result['CLEAR']
        identification_quality = result['Identity']
        VACE = result['VACE']
        count = result['COUNT']
        
        # Render the charts
        width_scale = 200
        width_ratio = [len(tracking_quality), len(detection_quality), len(identification_quality), len(VACE), len(count)]
        # Display the evaluation results
        tracking_chart = create_bar_chart(tracking_quality, "Tracking Quality (HOTA)", "#5470C6")
        detection_chart = create_bar_chart(detection_quality, "Detection Quality (CLEAR)", "#91CC75")
        identification_chart = create_bar_chart(identification_quality, "Identification Quality (Identity)", "#EE6666")
        VACE_chart = create_bar_chart(VACE, "VACE", "#fac858")
        count_chart = create_bar_chart(count, "Count", "#73c0de")
        
        components.html(tracking_chart, height=600)
        components.html(detection_chart, height=600)
        components.html(identification_chart, height=600)
        components.html(VACE_chart, height=600)
        components.html(count_chart, height=600)
        
        # Tell the user the evaluation is done
        st.success("Evaluation done! You can do new evaluation now.")

if st.button('Clear Evaluation'):
    # delete the saved file
    if my_upload is not None:
        os.remove(my_upload.name)
        # delete every image in the GT_plot and your_model_plot folder
        for file in os.listdir('GT_plot'):
            os.remove(f'GT_plot/{file}')
        for file in os.listdir('your_model_plot'):
            os.remove(f'your_model_plot/{file}')
        st.success(f'the txt file uploaded for video sequence {video_sequence} has been deleted')















