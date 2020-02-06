import os
import pickle
import numpy as np
from joint_tracking import joint_tracking
import math
from scipy.ndimage.filters import gaussian_filter
import cv2
import util
import time


def draw_boxes(input_image, frame_data, no_objects, colors):
    canvas = input_image.copy()
    colors = []
    for i, c in enumerate(colors):
        cur_canvas = canvas.copy()
        object_bbox = tuple(frame_data[frame_data[:, -1].astype(int) == i, 1:5].astype(int))
        cv2.rectangle(cur_canvas, object_bbox, color=c,
                      thickness=3)
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def draw_joints(input_image, all_peaks, subset, candidate, resize_fac=1):
    canvas = input_image.copy()

    for i in range(18):
        for j in range(len(all_peaks[i])):
            a = all_peaks[i][j][0] * resize_fac
            b = all_peaks[i][j][1] * resize_fac
            cv2.circle(canvas, (a, b), 2, util.colors[i], thickness=-1)

    stickwidth = 4

    for i in range(17):
        for s in subset:
            index = s[np.array(util.limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            y = candidate[index.astype(int), 0]
            x = candidate[index.astype(int), 1]
            m_x = np.mean(x)
            m_y = np.mean(y)
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(m_y * resize_fac), int(m_x * resize_fac)),
                                       (int(length * resize_fac / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, util.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def color_generator(n):
    colors = []
    for i in range(n):
        colors.append((np.random.randint(255), np.random.randint(255), np.random.randint(255)))
    return colors

def create_video(joints, tracks):
    print('start processing...')
    frame_rate_ratio = 3

    video = os.path.basename(VIDEO_FILE_PATH).split('.')[0]

    # Output location
    output_path = OUT_VIDEO_PATH
    output_format = '.mp4'
    video_output = output_path + video + str(start_datetime) + output_format

    # Video reader
    cam = cv2.VideoCapture(video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, output_fps, (orig_image.shape[1], orig_image.shape[0]))

    # object (person) tracking
    no_objects = int(tracks[:, -1].max()) + 1
    colors = color_generator(no_objects)

    i = 0  # input video frame id
    j = 0  # analyzed frames id
    while (cam.isOpened()) and ret_val is True and i < ending_frame:
        if i % frame_rate_ratio == 0:
            input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

            tic = time.time()

            # generate image with body parts
            body_parts, all_peaks, subset, candidate = joints[j, 1:5]
            canvas = draw_joints(orig_image, all_peaks, subset, candidate)

            # draw bounding boxes based on data analyzed by joint_tracking.py
            frame_data = tracks[tracks[:, 0].astype(int) == i, :]
            canvas = draw_boxes(canvas, frame_data, no_objects, colors)
            print('Processing frame: ', i)
            toc = time.time()
            print('processing time is %.5f' % (toc - tic))

            out.write(canvas)
            j += 1
        ret_val, orig_image = cam.read()

        i += 1
