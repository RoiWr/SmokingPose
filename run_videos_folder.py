import os
import sys
import argparse
import cv2
import time
import pickle
import glob
from config_reader import config_reader

from processing import extract_parts, draw

from model.cmu_model import get_testing_model

KERAS_WEIGHTS_FILEPATH = '/data/SMOKING_POSE/weights/model.h5'

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)

def get_pose(video_file, output_path):
    # Video input
    video_file = video_dir + video_file
    # data out
    output_data = []
    # Output location
    output_format = '.mp4'
    video_output = output_path + '/' + video + str(start_datetime) + output_format
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

    # scale_search = [1, .5, 1.5, 2]  # [.5, 1, 1.5, 2]
    # scale_search = scale_search[0:process_speed]
    #
    # params['scale_search'] = scale_search

    i = 0  # default is 0
    while(cam.isOpened()) and ret_val is True and i < ending_frame:
        if i % frame_rate_ratio == 0:

            input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

            tic = time.time()

            # generate image with body parts
            body_parts, all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
            output_data.append({'frame_id': i, 'body_parts': body_parts, 'all_peaks': all_peaks,
                              'subset': subset, 'candidate': candidate})
            canvas = draw(orig_image, all_peaks, subset, candidate)

            print('Processing frame: ', i)
            toc = time.time()
            print('processing time is %.5f' % (toc - tic))

            out.write(canvas)

        ret_val, orig_image = cam.read()

        i += 1

    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='input video directory name')
    # parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    # parser.add_argument('--process_speed', type=int, default=4,
    #                     help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')

    args = parser.parse_args()
    video_dir = args.dir
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    ending_frame = args.end

    # make save folders
    save_dir = video_dir + '_pose'
    video_save_dir = os.path.join(save_dir, 'videos')
    data_save_dir = os.path.join(save_dir, 'data')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        os.mkdir(video_save_dir)
        os.mkdir(data_save_dir)

    print('start processing...')

    # load model
    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(KERAS_WEIGHTS_FILEPATH)

    # load config
    params, model_params = config_reader()

    # go over files in video_dir
    filelist = glob.glob(video_dir + r'/*.mp4')
    for video in filelist:
        try:
            print('----------------------------------------')
            print(f'Process file {video}')
            output_data = get_pose(video, video_save_dir)
        except Exception as error:
            print(f"Error encountered in file {video}")
            print(error)
            continue

        # save data per file
        datafile_path = os.path.join(data_save_dir, video + '.pkl')
        with open(datafile_path, 'wb') as file:
            pickle.dump(output_data, file)

        print(f'Processed file {video} successfully')
        print('----------------------------------------')



