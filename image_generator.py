'''
This file contains the functions to generate the image visualization of the ground truth and the model outcome.
'''

import streamlit as st
from PIL import Image
from io import BytesIO
from pyecharts.charts import Bar
from pyecharts import options as opts
import streamlit.components.v1 as components
import cv2
import numpy as np
from pathlib import Path
import base64

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

# Function to create a zoomable image viewer in Streamlit
def create_zoomable_image(image_path, col, frame_num):
    # Convert image to base64 to embed in HTML
    encoded_image = base64.b64encode(open(image_path, "rb").read()).decode()
    html_code = f"""
    <link rel="stylesheet" href="https://unpkg.com/viewerjs/dist/viewer.min.css">
    <script src="https://unpkg.com/viewerjs/dist/viewer.min.js"></script>
    <div style="text-align: center; width: 384px; height: 216px; overflow: hidden; display: flex; justify-content: center; align-items: center;">
        <img id="image" src="data:image/jpg;base64,{encoded_image}" style="max-width: 100%; max-height: 100%; object-fit: contain;">
    </div>
    <script>
        var image = document.getElementById('image');
        var viewer = new Viewer(image, {{
            inline: true,
            navbar: false,
            toolbar: {{
                zoomIn: 4,
                zoomOut: 4,
                oneToOne: 1,
                reset: 1,
                prev: 0,
                play: 0,
                next: 0,
                rotateLeft: 4,
                rotateRight: 4,
                flipHorizontal: 4,
                flipVertical: 4
            }},
            viewed() {{
                viewer.zoomTo(0.22); // Adjust this value as needed to ensure the full image is visible
            }}
        }});
    </script>
    """
    with col:
        components.html(html_code, height=216, width=384)
        st.write(f"frame {frame_num}")

def generate_image_zoomable(my_upload, video_sequence, values, col1, col2):
    text_file_path = f"data/gt/mot_challenge/MOT16-train/{video_sequence}/gt/gt.txt"
    img_path_base = f"MOT16/train/{video_sequence}/img1/"
    
    # Process the first frame for ground truth and model outcome
    img_path_0 = f"{img_path_base}{str(values[0]).zfill(6)}.jpg"
    draw_box(text_file_path, img_path_0, values[0], 'GT_plot')
    col1.write("Ground Truth MoT")
    create_zoomable_image(f'GT_plot/output_image_{values[0]}.jpg', col1, values[0])
    draw_box(my_upload.name, img_path_0, values[0], 'your_model_plot')
    col2.write("Your Model MoT")
    create_zoomable_image(f'your_model_plot/output_image_{values[0]}.jpg', col2, values[0])
    
    # Process the second frame for ground truth and model outcome
    img_path_1 = f"{img_path_base}{str(values[1]).zfill(6)}.jpg"
    draw_box(text_file_path, img_path_1, values[1], 'GT_plot')
    create_zoomable_image(f'GT_plot/output_image_{values[1]}.jpg', col1, values[1])
    draw_box(my_upload.name, img_path_1, values[1], 'your_model_plot')
    create_zoomable_image(f'your_model_plot/output_image_{values[1]}.jpg', col2, values[1])

#Takes a filename and a frame id, returns all bounding boxes and scores in that frame.
#tid = target ID
def get_bbox(file, frame_id):
    frame_ids = []
    tids = []
    tlwhs_lst = []
    scores = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            values = line.split(',')
            if (not values[0].isdigit()) or int(values[0])!=frame_id:
                continue

            frame_id = int(values[0])
            tid = int(float(values[1]))
            tlwhs = []
            for i in values[2:6]:
                tlwhs.append(float(i))
            score = float(values[6])
            frame_ids.append(frame_id)
            tids.append(tid)
            tlwhs_lst.append(tlwhs)
            scores.append(score)
    #All return types are lists
    return frame_ids, tids, tlwhs_lst, scores

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

#plot bounding boxes on a single image
def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    # frame_id = frame_id+1
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def draw_box(bbox, img, frame_id, folder):
    frame_ids, tids, tlwhs_lst, scores = get_bbox(bbox, frame_id)
    if len(frame_ids)>0:
        image = plot_tracking(img, tlwhs_lst, tids, frame_id=frame_ids[0])
        output_path = f'{folder}/output_image_{frame_id}.jpg'
        cv2.imwrite(output_path, image)