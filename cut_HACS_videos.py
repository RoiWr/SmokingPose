'''Script to cut the downloaded clips of the HACS dataset to the annotated labeled 2 sec clips
Roi Weinberger & Sagiv Yaarri. Date: 1/2/2020 '''

import os
import numpy as np
import pandas as pd
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# TODO: add logging

# CONSTANTS
ANNOTATION_FILE_PATH = '/data/smoking_pose/HACS/annotations/HACS-dataset/HACS_v1.1.1/HACS_clips_v1.1.1.csv'
VIDEO_DIR = '/data/smoking_pose/HACS'
SAVE_DIR = '/data/smoking_pose/HACS/clips'
DEF_CATEGORY = 'Smoking a cigarette'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default=DEF_CATEGORY, help='category of videos to cut to clips')

    args = parser.parse_args()
    category = args.category

    # load annotations csv and subset to category
    df = pd.read_csv(ANNOTATION_FILE_PATH)
    df_cat = df.loc[df.classname == category, :]
    del df

    labels = []
    for row in df_cat.iterrows():
        category_dir = category.replace(' ', '_')
        filename = os.path.join(VIDEO_DIR, category_dir, row['youtube_id'] + '.mp4') # TODO: check that '//' work in bash
        # TODO: logger
        if os.path.isfile(filename):
            clip = ffmpeg_extract_subclip(filename, row['start'], row['end'], targetname=None)
            # TODO: save clip to SAVE_DIR
            labels.append({'youtube_id': row['youtube_id'], 'label': row['label']})
        else:
            print(f'Video {row.youtube_id} not found. skip to next video')  # TODO: do in logger
            continue

    # TODO: save labels as csv
    labels_df = pd.DataFrame(labels)
    save_filepath = os.path.join(VIDEO_DIR, category_dir, 'labels.csv')
    with ('save_filepath', 'w') as file:
