'''Script to cut the downloaded clips of the HACS dataset to the annotated labeled 2 sec clips
Roi Weinberger & Sagiv Yaarri. Date: 1/2/2020 '''

import os
import pandas as pd
import argparse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# CONSTANTS
ANNOTATION_FILE_PATH = '/data/smoking_pose/HACS/HACS-dataset/HACS_v1.1.1/HACS_clips_v1.1.1.csv'
VIDEO_DIR = '/data/smoking_pose/HACS'
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

    # create save folder
    category_dir = category.replace(' ', '_')
    save_path = os.path.join(VIDEO_DIR, 'clips', category_dir)
    if not os.path.isdir(save_path):
        os.mkdir(os.path.join(VIDEO_DIR, 'clips'))
        os.mkdir(save_path)

    labels = []
    for i, row in df_cat.iterrows():
        filename = f'v_{row.youtube_id}.mp4'
        filepath = os.path.join(VIDEO_DIR, category_dir, filename) # TODO: check that '//' work in bash

        if os.path.isfile(filepath):
            outfilename = f'v_{row.youtube_id}_{row.start:.0f}_{row.end:.0f}.mp4'
            outfile_path = os.path.join(save_path, outfilename)
            ffmpeg_extract_subclip(filepath, row['start'], row['end'], targetname=outfile_path)
            labels.append({'youtube_id': row['youtube_id'], 'category': category, 'label': row['label']})
            print(f'Clipped video {filename} successfully')
        else:
            print(f'Video {row.youtube_id} not found. skip to next video')
            continue

    labels_df = pd.DataFrame(labels)
    save_filepath = os.path.join(save_path, 'labels.csv')
    labels_df.to_csv(save_filepath)
    print(f'Labels file saved')


