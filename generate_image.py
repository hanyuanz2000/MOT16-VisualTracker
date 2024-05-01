import streamlit as st
from PIL import Image
from io import BytesIO
from pyecharts.charts import Bar
from pyecharts import options as opts
import streamlit.components.v1 as components
from plot_tracking import * 

# ===========================
def image_check(my_upload, MAX_FILE_SIZE):
    if my_upload:
         if my_upload.size > MAX_FILE_SIZE:
            st.sidebar.error(f"File size exceeds {MAX_FILE_SIZE}/{MAX_FILE_SIZE/1024/1024} MB")
        # else:
            # fix_image(upload=my_upload)
    # else:
        # fix_image("./zebra.jpg")
        
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# upload:txt file, video_no: video, values: start and end frame
def generate_image(my_upload, video_sequence, values, col1, col2):
    
    text_file_path = f"MOT16/train/{video_sequence}/det/det.txt"
    img_path = f"MOT16/train/{video_sequence}/img1/{str(values[0]).zfill(6)}.txt"

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

