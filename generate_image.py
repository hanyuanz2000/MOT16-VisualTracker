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
            return False
        else:
            return True    

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# upload:txt file, video_no: video, values: start and end frame
def generate_image(my_upload, video_sequence, values, col1, col2):
    # text_file_path = f"MOT16/train/{video_sequence}/det/det.txt"
    text_file_path = f"data/gt/mot_challenge/MOT16-train/{video_sequence}/gt/gt.txt"
    img_path = f"MOT16/train/{video_sequence}/img1/{str(values[0]).zfill(6)}.jpg"

    # plot the ground truth
    col1.write("Ground Truth MoT")
    draw_box(text_file_path, img_path, values[0], 'GT_plot')
    
    draw_box(text_file_path, img_path, values[1], 'GT_plot')
    image_gt_0 = Image.open('GT_plot/output_image_'+str(values[0])+'.jpg')   
    col1.image(image_gt_0, use_column_width=True)
    col1.write(f"frame {str(values[0])}")

    image_gt_1 = Image.open('GT_plot/output_image_'+str(values[1])+'.jpg')   
    col1.image(image_gt_1, use_column_width=True)
    col1.write(f"frame {str(values[1])}")

    # plot the model outcome
    col2.write("Your Model MoT")
    draw_box(my_upload.name, img_path, values[0], 'your_model_plot')
    draw_box(my_upload.name, img_path, values[1], 'your_model_plot')
    image_your_model_0 = Image.open('your_model_plot/output_image_'+str(values[0])+'.jpg')   
    col2.image(image_your_model_0, use_column_width=True)
    col2.write(f"frame {str(values[0])}")

    image_your_model_1 = Image.open('your_model_plot/output_image_'+str(values[1])+'.jpg')   
    col2.image(image_your_model_1, use_column_width=True)
    col2.write(f"frame {str(values[1])}")

    st.sidebar.markdown("\n")
    # st.sidebar.download_button("Download visualization", convert_image(fixed), "fixed.png", "image/png")

