# -*- coding: utf-8 -*-

from fastai.conv_learner import *
import os
import nmslib
import time
import cv2
import numpy as np
import fastText as ft
import urllib.request
import zipfile
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


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
    t = time.time()
    print('Ordering frame indices ... ')
    for video in results.keys():
        indices_cos = list(zip(results[video][0], results[video][1]))
        indices_cos.sort(key=lambda x: x[0])
        current_index = indices_cos[0][0]
        indices[video] = []
        count = 0
        for i, tup in enumerate(indices_cos):
            if indices_cos[i][1] > cosine_threshold:
                continue
            elif (len(indices[video]) == 0):
                indices[video].append([indices_cos[i][0]])
                continue
            elif indices_cos[i][0] != current_index + 1:
                indices[video].append([indices_cos[i][0]])
                count += 1
            else:
                indices[video][count].append(indices_cos[i][0])
            current_index = indices_cos[i][0]
    print('Done ordering frame indices ... {} seconds'.format(time.time() - t))
    return indices


def frames_to_videos(original_videos_path, frame_indices_dict, results_clips_path):
    """
    :param original_videos_path: path to the videos
    :param frame_indices_dict: indices of the frames as a result of the search
    :param results_clips_path: path to the directory where the resulting videos are saved
    :return:
    """
    print('Constricting video clips ...')
    t = time.time()

    for video in frame_indices_dict.keys():
        if frame_indices_dict[video] == []:
            continue
        else:
            count = 0
            if not os.path.exists(os.path.join(results_clips_path, video)):
                os.mkdir(os.path.join(results_clips_path, video))
                for i, item in enumerate(frame_indices_dict[video]):
                    target_path = os.path.join(results_clips_path, video, "subvideo_{}.mp4".format(count))
                    start_time = item[0]
                    end_time = item[-1]
                    ffmpeg_extract_subclip(os.path.join(original_videos_path, "{}".format(video)),
                                           start_time - 1,
                                           end_time + 1,
                                           targetname=target_path)
                    count += 1
            elif os.path.exists(os.path.join(results_clips_path, video)) and len(
                    os.listdir(os.path.join(results_clips_path, video))) > 0:
                print("{} directory exists and is not empty".format(os.path.join(results_clips_path, video)))
        print('Done constructing video clips ... {} seconds'.format(time.time() - t))


class Catalog:
    def __init__(self, path_to_catalog):
        self.catalog_path = path_to_catalog
        basedir = os.path.abspath(os.path.dirname(__file__))
        self.model_path = os.path.join(basedir, '../data/model')
        self.word_vec_path = os.path.join(basedir, '../data/word_vecs')
        self.word_vecs = os.path.join(self.word_vec_path, 'wiki.en.bin')
        self.results_path = os.path.join(self.catalog_path, 'results')
        self.index_path = os.path.join(self.catalog_path, 'indexes')

        if os.path.exists(self.model_path, 'images.pkl.zip'):
            zipfile_path = os.path.join(self.model_path, 'images.pkl.zip')
            zip_ref = zipfile.ZipFile(zipfile_path, 'r')
            zip_ref.extractall(self.model_path)
            zip_ref.close()
            os.remove(zipfile_path)
        if os.path.exists(self.model_path, 'img_vecs.pkl.zip'):
            zipfile_path = os.path.join(self.model_path, 'img_vecs.pkl.zip')
            zip_ref = zipfile.ZipFile(zipfile_path, 'r')
            zip_ref.extractall(self.model_path)
            zip_ref.close()
            os.remove(zipfile_path)    
        self.images_pkl = pickle.load(open(os.path.join(self.model_path, 'images.pkl'), 'rb'))
        self.image_vec_pkl = pickle.load(open(os.path.join(self.model_path, 'img_vecs.pkl'), 'rb'))

        self.fake_path = os.path.join(self.catalog_path, 'tmp')

        if not os.path.isdir(self.word_vec_path):
            print("Downloading fastText word-vectors file ~ 9GB ...")
            os.mkdir(self.word_vec_path)
            zipfile_path = os.path.join(self.word_vec_path, 'wiki.en.zip')
            urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip', zipfile_path)
            print("\tUnzipping fastText word-vectors file ~ 9GB ...")
            zip_ref = zipfile.ZipFile(zipfile_path, 'r')
            zip_ref.extractall(self.word_vec_path)
            zip_ref.close()
            print("\tRemoving fastText word-vectors file ~ 9GB ...")
            os.remove(zipfile_path)
            os.remove(os.path.join(self.word_vec_path, 'wiki.en.vec'))

    def index(self, forced=False):
        """Index catalog."""
        print("Initializing the model and indexing your catalog ...")
        if os.path.isdir(self.index_path):
            print("\tWarning: Index exist.")
            if forced:
                print("\tModel will be re-initialized and catalog will be re-indexed.")
                shutil.rmtree(self.index_path)
                self.__model_init()
                self.__create_save_index()
            else:
                print("\tModel and catalog will be re-used.")
        else:
            self.__model_init()
            self.__create_save_index()

    def search_text(self, text):
        """Search text in the catalog."""

        print('\tSearching ...')
        # Delete the last search results and create a new results directory
        if os.path.exists(self.results_path):
            shutil.rmtree(self.results_path)
        os.mkdir(self.results_path)
            
        index = self.__load_index()

        # Grab the word vector of the input text
        en_vecd = ft.load_model(os.path.join(self.word_vec_path, 'wiki.en.bin'))
        word_vecs = []
        for word in text:
            word_vecs.append(en_vecd.get_word_vector(word))
        word_vec = np.mean(word_vecs, axis=0)
        
        start_time = time.time()
#        # Search for images
#         idxs, dists = self.__get_knn(index['photos'][1], word_vec)
#         for i, cos in zip(idxs, dists):
#             #filename = os.path.join(self.results_path, "{}.png".format(round(cos, 2)))
#             filename = os.path.join(self.results_path, "{}".format(index['photos'][0][i]))
#             print(filename)
#             print(index['photos'][0][i])
#             cv2.imwrite(filename, index['photos'][0][i])
                
        # Search for videos
        results_vecs = {}
        for key in index.keys():
            if key == 'photos':
                continue
            results_vecs[key] = self.__get_knn(index[key], word_vec)
        frames_ordered = order_frame_indices(results_vecs, cosine_threshold=0.7)
        frames_to_videos(self.catalog_path, frames_ordered, self.results_path)

        print("\tDone searching! ... {} seconds".format(time.time() - start_time))

    def __get_knn(self, index, vec):
        return index.knnQuery(vec, k=10)

    def __touch(self, path):
        """Create imagenet directory structure (with fake .JPEG files)."""
        with open(path, 'a'):
            os.utime(path, None)

    def __create_tmp(self):
        """
        Create tmp folder. Create train and valid inside tmp.
        Create a directory for every category inside train and valid.
        """
        for image in self.images_pkl:
            image_path = '/'.join(image.split('/')[:-1])
            if not os.path.exists(os.path.join(self.fake_path, image_path)):
                os.makedirs(os.path.join(self.fake_path, image_path))

    def __create_tmpfiles(self):
        """Create .JPEGs corresponding to the images used during the training time."""
        for image in self.images_pkl:
            if not os.path.exists(os.path.join(self.fake_path, image)):
                self.__touch(os.path.join(self.fake_path, image))

    def __model_init(self):
        """Initialize the model."""
        print("\tInitializing the model ...")
        arch = resnet50
        n = len(self.images_pkl)
        n_val = 27455
        val_idxs = list(range(n - n_val, n))

        self.__create_tmp()
        self.__create_tmpfiles()

        tfms = tfms_from_model(arch, 224, transforms_side_on, max_zoom=1.1)
        self.md = ImageClassifierData.from_names_and_array(self.fake_path,
                                                           self.images_pkl,
                                                           self.image_vec_pkl,
                                                           val_idxs=val_idxs,
                                                           classes=None,
                                                           tfms=tfms,
                                                           continuous=True,
                                                           bs=256)

        models = ConvnetBuilder(arch,
                                300,
                                is_multi=False,
                                is_reg=True,
                                xtra_fc=[1024],
                                ps=[0.2, 0.2])

        self.learn = ConvLearner(self.md, models)  # , precompute=True)
        self.learn.opt_fn = partial(optim.Adam, betas=(0.9, 0.99))

        self.learn.load(os.path.join(self.model_path, 'pre0'))

    def __create_index(self, video_features):
        index = nmslib.init(space='angulardist')
        index.addDataPointBatch(video_features)
        index.createIndex()
        return index

    def __create_save_index(self):
        """Create an index."""
        index = {}
        image_paths = []
        image_pred_wv = []
        basedir = os.path.abspath(os.path.dirname(__file__))
        os.mkdir(self.index_path)

        print('\tCreating the index ...')
        start_time = time.time()
        for f in os.listdir(self.catalog_path):
            if '.mp4' in f:
                t = time.time()
                print('\tExtracting frames from {}'.format(f))
                video_features = []

                cap = cv2.VideoCapture(os.path.join(self.catalog_path, f))
                frame_rate = cap.get(5)
                while cap.isOpened():
                    frame_id = cap.get(1)
                    ret, frame = cap.read()
                    if ret != True:
                        break
                    if frame_id % round(frame_rate) == 0:
                        filename = os.path.join(basedir, "frame_tmp.png")
                        cv2.imwrite(filename, frame)
                        img = open_image(os.path.join(basedir, "frame_tmp.png"))
                        t_img = self.md.val_ds.transform(img)
                        pred = self.learn.predict_array(t_img[None])
                        video_features.append(pred)
                cap.release()

                index[f] = self.__create_index(video_features)
                os.remove(os.path.join(basedir, "frame_tmp.png"))
                print(
                    "\tFinished extracting frames from {} .......... {} seconds\n".format(f, round(time.time() - t, 2)))

            elif '.jpg' in f:
                if 'photos' not in index.keys():
                    index['photos'] = []
                img = open_image(os.path.join(self.catalog_path, f))
                t_img = self.md.val_ds.transform(img)
                pred = self.learn.predict_array(t_img[None])
                image_pred_wv.append(pred)
                image_paths.append(os.path.join(self.catalog_path, f))
            else:
                continue

        index['photos'] = (image_paths, self.__create_index(image_pred_wv))
        print(index)
        # Save index to disc
        print('\tSaving the index ...')
        for key in index.keys():
            if key == 'photos':
                index[key][1].saveIndex(os.path.join(self.index_path, '{}.nmslib'.format(key)), save_data=True)
                pickle.dump(index[key][0], open(os.path.join(self.index_path, '{}.pkl'.format(key)), 'wb'))
                #print('Load')
                #cur_index = nmslib.init(space='angulardist')
                #cur_index.loadIndex(os.path.join(self.index_path, 'photos.nmslib'), load_data=True)
            else:
                index[key].saveIndex(os.path.join(self.index_path, '{}.nmslib'.format(key)), save_data=True)
        print('\tDone creating and saving the index ... {} seconds'.format(time.time() - start_time))

    def __load_index(self):
        """Load previously saved index"""
        index = {}
        index_path = os.path.join(self.catalog_path, 'indexes')

        for f in os.listdir(index_path):
            if 'dat' in f:
                key = f.split('.nmslib')[0]
                # print(key)
                cur_index = nmslib.init(space='angulardist')
                cur_index.loadIndex(os.path.join(index_path, f.split('.dat')[0]), load_data=True)
                if 'photo' in f:
                    image_paths = pickle.load(open(os.path.join(index_path, 'photos.pkl'), 'rb'))
                    index[key] = (image_paths, cur_index)
                else:
                    index[key] = cur_index
        return index
