import cv2
import math
import os
import time
import matplotlib.pyplot as plt
import youtube_dl
import io
import base64
from IPython.display import HTML
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def dwl_vid():
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([zxt])

def download_youtube_video():
    channel = 1
    while (channel == int(1)):
        link_of_the_video = input("Copy & paste the URL of the YouTube video you want to download:- ")
        zxt = link_of_the_video.strip()
        dwl_vid()
        channel = int(input("Enter 1 if you want to download more videos \nEnter 0 if you are done "))

def videos_to_frames(video_dir_path):
    """
    Extract 1 frame per second for every video in the directory
    param video_dir_path: path to the videos
    """
    start_time = time.time()
    for videofile in os.listdir(video_dir_path):
        if videofile.split(".")[1] != 'mp4':
            continue
        elif not os.path.exists(os.path.join(video_dir_path, "/frames/{}".format(videofile.split(".")[0]))):
            start_time = time.time()
            os.mkdir(os.path.join(video_dir_path, "/frames/{}".format(videofile.split(".")[0])))
            count = 0
            print('Extracting frames from {}'.format(videofile))
            cap = cv2.VideoCapture(videofile)  # capturing the video from the given path
            frameRate = cap.get(5)  # frame rate
            x = 1
            while (cap.isOpened()):
                frameId = cap.get(1)  # current frame number
                ret, frame = cap.read()
                if (ret != True):
                    break
                if (frameId % math.floor(frameRate) == 0):
                    filename = "../data/train/frames/{}/frame{}.png".format(videofile.split(".")[0], count)
                    count += 1
                    cv2.imwrite(filename, frame)
            cap.release()
            print("Finished extracting frames from {}! .......... {} seconds".format(videofile, round(time.time()-start_time, 2)))
        elif os.listdir(os.path.join(video_dir_path, "/frames/{}".format(videofile.split(".")[0]))) > 0:
            print("The frames have already been extracted")

def plot_image(image_path):
    img = plt.imread(image_path)
    plt.imshow(img)

def show_video(video_path):
    video = io.open(video_path, 'r+b').read()
    encoded = base64.b64encode(video)
    HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))

def frames_to_videos(original_video_path, frame_indices, destination_path):
    t = time.time()
    count = 0
    for i, item in enumerate(frame_indices):
        if not os.path.exists(destination_path):
            os.mkdir(os.path.join(destination_path, "/subvideo_{}.mp4".format(count)))
            start_time = item[0]
            end_time = item[-1]
            if end_time - start_time > 1:
                ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=target_name)
                count += 1
            else:
                continue
    print("Finished constructing video clips ......... {} seconds".format(time.time() - t))

def order_frame_indices(results_indices):
    results_indices.sort(key=lambda x: x[0])
    current_index = results_indices[0][0]
    indices = []
    count = 0
    for i, list_item in enumerate(results_indices):
        if (len(indices) == 0):
            indices.append([list_item[0]])
            continue
        elif list_item[0] != current_index + 1:
            indices.append([list_item[0]])
            count += 1
        else:
            indices[count].append(list_item[0])
        current_index = results_indices[i][0]
    return indices

def plot_frames(n_frames, frames_path, indices):
    fig = plt.figure(figsize=(40, 40))
    for i in range(1, n_frames + 1):
        img = plt.imread(os.path.join(frames_path, '/frame{}.png'.format(indices[i - 1][0]))
        fig.add_subplot(n_frames/3, 3, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
