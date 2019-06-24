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


def videos_to_frames(video_dir_path, frames_dir_path):
    """
    Extract 1 frame per second for every video in the directory
    :param video_dir_path: path to the videos
    :param frames_dir_path: path to the frames directory where the extracted frames will be saved
    :return:
    """
    start_time = time.time()
    for videofile in os.listdir(video_dir_path):
        video_name = videofile.split(".")[0]

        if videofile.split(".")[1] != 'mp4':
            print('Ecpected .mp4, received {}'.format(videofile.split(".")[1]))
            continue
        elif not os.path.exists(os.path.join(frames_dir_path, video_name)):
            start_time = time.time()
            os.mkdir(os.path.join(os.path.join(frames_dir_path, video_name)))
            print('Extracting frames from {}'.format(videofile))
            cap = cv2.VideoCapture(os.path.join(video_dir_path, videofile))  # capturing the video from the given path
            frameRate = cap.get(5)                                           # frame rate
            count = 0

            while (cap.isOpened()):
                frameId = cap.get(1)                                         # current frame number
                ret, frame = cap.read()
                if (ret != True):
                    break
                if (frameId % round(frameRate) == 0):
                    filename = os.path.join(frames_dir_path, video_name, "frame{}.png".format(count))
                    count += 1
                    cv2.imwrite(filename, frame)
            cap.release()
            print("Finished extracting frames from {} .......... {} seconds\n".format(videofile, round(time.time() - start_time, 2)))
        elif len(os.listdir(os.path.join(frames_dir_path, video_name))) > 0:
            print("The frames have already been extracted\n")


def plot_image(image_path):
    """
    Show the image
    :param image_path: path to the image to be shown
    :return: plot
    """
    img = plt.imread(image_path)
    plt.imshow(img)


def plot_similar_frames(frames_dir_path, video_name, results, n_frames=12):
    """
    Plot frames (images)
    :param frames_dir_path: path (string) to the directory of the frames
    :param video_name: name (string) of the video the frames of which are going to be plotted
    :param results: Dictionary where keys are names of the videos (string) and
                    values are lists of lists containing the frame number (int), frame path (string), and frame cosine (float)
    :param n_frames: number of frames (int) to be shown. NOTE: this is a multiple of 3. Tested well for n_frames=3, 6, 9, and 12
                     Typically corresponds to the number of frames that we want to search for
    :return:
    """
    if n_frames % 3 != 0:
        print("Warning: n_frames should be a multiple of 3")
    else:
        fig = plt.figure(figsize=(n_frames * 6, n_frames * 3))
        for i in range(1, n_frames + 1):
            img = plt.imread(os.path.join(frames_dir_path, video_name, 'frame{}.png'.format(results[video_name][i - 1][0])))
            fig.add_subplot(n_frames / 3, 3, i)
            plt.imshow(img)
            plt.title("frame{}\nCosine similarity: {}".format(results[video_name][i-1][0],
                                                              round(float(results[video_name][i - 1][2]), 4)), size=20 + n_frames * 3)
            plt.axis('off')
            plt.tight_layout()
        plt.show()


def order_frame_indices(results, cosine_threshold=0.7):
    """
    Order the indices of the frames given the search results
    :param results: Dictionary where keys are names of the videos (string) and
                    values are lists of lists containing the frame number (int), frame path (string), and frame cosine (float)
    :param cosine_threshold: Frames whose cosine with is bigger than this threshold will be excluded
    :return: dictionary where keys are video names and values are lists of lists. Every inner list contains consecutive frame numbers
    >>> order_frame_indices({"video1":[[1, "frame1_path", 0.1], [11, "frame10_path", 0.1], [10, "frame10_path", 0.6], [12, "frame12_path", 0.9]],
    ...                      "video2": [[9, "frame9_path", 0.2], [2, "frame2_path", 0.5], [4, "frame4_path", 0.8]]})
    {'video1': [[1], [10, 11]], 'video2': [[2], [9]]}
    """
    indices = {}
    for video in results.keys():
        results[video].sort(key=lambda x: x[0])
        current_index = results[video][0][0]
        indices[video] = []
        count = 0
        for i, list_item in enumerate(results[video]):
            if results[video][i][2] > cosine_threshold:
                continue
            elif (len(indices[video]) == 0):
                indices[video].append([list_item[0]])
                continue
            elif list_item[0] != current_index + 1:
                indices[video].append([list_item[0]])
                count += 1
            else:
                indices[video][count].append(list_item[0])
            current_index = results[video][i][0]
    return indices


def frames_to_videos_2(original_videos_path, frame_indices_dict, results_clips_path):
    """
    :param original_video_path:
    :param frame_indices:
    :param destination_path:
    :return:
    """
    for video in frame_indices_dict.keys():
        if frame_indices_dict[video] == []:
            continue
        else:
            count = 0
            if not os.path.exists(os.path.join(results_clips_path, video)):
                os.mkdir(os.path.join(results_clips_path, video))
                for i, item in enumerate(frame_indices_dict[video]):
                    target_path = os.path.join(results_clips_path, video, "subvideo_{}.mp4".format(count))
                    #os.mkdir(target_path)
                    start_time = item[0]
                    end_time = item[-1]
                    if end_time - start_time > 1:
                        print("\n\n", start_time, end_time)
                        ffmpeg_extract_subclip(os.path.join(original_videos_path, "{}.mp4".format(video)), start_time-1, end_time+1, targetname=target_path)
                        count += 1
                    else:
                        print(start_time, end_time, 'continue')
                        continue
            elif os.path.exists(os.path.join(results_clips_path, video)) and len(os.listdir(os.path.join(results_clips_path, video))) > 0:
                print("{} directory exists and is not empty".format(os.path.join(results_clips_path, video)))


def cv2_video_analysis(video_path):
    """
    print several features of the given video
    :param video_path: path to the video
    :return: print frames per second, number of total frames, duration of the video in seconds and minutes:seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    minutes = int(duration / 60)
    seconds = duration % 60

    print('fps = ' + str(fps))
    print("cap.get(5) ...", cap.get(5))
    print('number of total frames = ' + str(frame_count))
    print('duration (seconds) = ' + str(duration))
    print('duration (minutes:seconds) = ' + str(minutes) + ':' + str(seconds))

    cap.release()
