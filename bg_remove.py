import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64
from pyecharts.charts import Bar
from pyecharts import options as opts
import streamlit.components.v1 as components
import cv2
from plot_tracking import * 

st.set_page_config(layout="wide", page_title="Vis Your MoT Model!")

st.write("## 👀Visualize your model performance")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


st.sidebar.write("## Upload and download :gear:")

if 'page' not in st.session_state:
    st.session_state.page = 1 

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload your model outcome", type=["txt"])


dict_video = {1: 450, 2: 600, 3: 1500, 4: 1050, 5: 837,  6: 1194, 7: 500, 8: 625, 9: 525, 10: 654, 11: 900, 12: 900, 13: 750, 14: 750}



video = st.sidebar.selectbox(
    'Video You Want To Explore',
    (f'Video{i}' for i in [2,4,5,9,10,11,13]),
    index=0,
)
no_frames = dict_video[int(video[5:])]

values = st.slider(
    'Select a range of values',
    1, no_frames, (1, no_frames), key='values'
)

st.write('Values:', values)

# upload:txt file, video_no: video, values: start and end frame
def generate_image(my_upload, video_no, values):
    
    text_file_path = f"/workspaces/Vis_MoT/MOT16/train/MOT16-{str(video_no[5:]).zfill(2)}/det/det.txt"
    img_path = f"/workspaces/Vis_MoT/MOT16/train/MOT16-{str(video_no[5:]).zfill(2)}/img1/{str(values[0]).zfill(6)}.txt"

    col1.write("Ground Truth MoT")
    draw_box(text_file_path, img_path, values[0])
    col1.write("\n")
    draw_box(text_file_path, img_path, values[1])
    # col1.image(image)

    # fixed = remove(image)
    col2.write("Your Model MoT")
    draw_box(my_upload, img_path, values[0])
    col2.write("\n")
    draw_box(my_upload, img_path, values[1])

    st.sidebar.markdown("\n")
    # st.sidebar.download_button("Download visualization", convert_image(fixed), "fixed.png", "image/png")

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    # else:
        # fix_image(upload=my_upload)
# else:
    # fix_image("./zebra.jpg")
if my_upload:
    generate_image(my_upload=my_upload, video_no=video, values = values)


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

