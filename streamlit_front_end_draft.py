import streamlit as st
from PIL import Image
from io import BytesIO
import base64
from pyecharts.charts import Bar
from pyecharts import options as opts
import streamlit.components.v1 as components
from plot_tracking import * 
from multiprocessing import Process, Value
import traceback
from werkzeug.utils import secure_filename
from multiprocessing import freeze_support
from filtered_frame_eval import run_evaluation
from generate_image import generate_image, image_check
import time

# ===========================Streamlit Setup============================
# Set page title and layout
st.set_page_config(layout="wide", page_title="Vis Your MoT Model!")
st.write("## 👀Visualize your model performance")

# Set max file size and acceptable file types
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Display the sidebar
st.sidebar.write("## Upload and download :gear:")

if 'page' not in st.session_state:
    st.session_state.page = 1 

# ===========================Select Video and Frame============================
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload your model outcome", type=["txt"])

# Video selection
dict_video_sequence = {'MOT16-02': 600, 'MOT16-04': 1050, 'MOT16-05': 837, 'MOT16-09': 525, 'MOT16-10': 654, 'MOT16-11': 900, 'MOT16-13': 750}
video_sequence = st.sidebar.selectbox(
    'Video Sequence You Want To Explore',
    (f'MOT16-02', f'MOT16-04', f'MOT16-05', f'MOT16-09', f'MOT16-10', f'MOT16-11', f'MOT16-13'),
    index=0,
)

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
image_check(my_upload, MAX_FILE_SIZE)
# generate_image(my_upload=my_upload, video_sequence = video_sequence, values = values, col1=col1, col2=col2)


# ===========================Evaluation============================
# Setup for managing the evaluation process
if 'eval_process' not in st.session_state:
    st.session_state['eval_process'] = None

# Function to handle the evaluation process
def eval_wrapper(t0, t1):
    with st.spinner("Evaluation ongoing for current selected frame..."):
        result = run_evaluation(t0, t1)
    st.success(f"Current evaluation results: {result}")

t0, t1 = values[0], values[1]
# Start Evaluation Button
if st.button("Start Evaluation"):
    # Check if a process is already running
    if st.session_state['eval_process'] is not None and st.session_state['eval_process'].is_alive():
        st.warning("An evaluation is already running. Please wait until it finishes.")
    else:
        # Stop and join any previous process that may not be alive
        if st.session_state['eval_process'] is not None:
            st.session_state['eval_process'].join()

        # Start a new evaluation process
        st.session_state['eval_process'] = Process(target=eval_wrapper, args=(t0, t1))
        st.session_state['eval_process'].start()

# ===========================Pyecharts Code============================
c = (Bar()
        # TODO: metric name
.add_xaxis(["Microsoft", "Amazon", "IBM", "Oracle", "Google", "Alibaba"])
    # TODO: metric value
.add_yaxis('2017-2018 Revenue in (billion $)', [21.2, 20.4, 10.3, 6.08, 4, 2.2])
.set_global_opts(title_opts=opts.TitleOpts(title="Top cloud providers 2018", subtitle="2017-2018 Revenue"),
                    toolbox_opts=opts.ToolboxOpts())
.render_embed() # generate a local HTML file
)
components.html(c, width=1000, height=1000)

















