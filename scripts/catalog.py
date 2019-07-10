# -*- coding: utf-8 -*-

from fastai.conv_learner import *
import os
import nmslib
import time
import cv2
import numpy as np
import fasttext as ft


class Catalog:
    def __init__(self, path_to_catalog):
        self.catalog_path = path_to_catalog
        basedir = os.path.abspath(os.path.dirname(__file__))
        self.model_path = os.path.join(basedir, '../data/model')

        self.images_pkl = pickle.load(open(os.path.join(self.model_path, 'images.pkl'), 'rb'))
        self.image_vec_pkl = pickle.load(open(os.path.join(self.model_path, 'img_vecs.pkl'), 'rb'))

        print(len(self.images_pkl))
        print(len(self.image_vec_pkl))

        self.fake_path = os.path.join(self.catalog_path, 'tmp')
        self.indexed = os.path.isdir(self.fake_path)

    def index(self, forced=False):
        """Index catalog."""
        print("Indexing your catalog ...")
        if os.path.isdir(self.fake_path):
            print("Warning: Index exist.")
            if forced:
                print("\tCatalog will be re-indexed.")
                shutil.rmtree(self.fake_path)
                self.__create_tmp()
                self.__create_tmpfiles()
            else:
                print("\tCatalog will be re-used.")
        else:
            self.__create_tmp()
            self.__create_tmpfiles()

        self.__model_init()

    def search_image(self, path_to_image):
        """Search image in the catalog."""

    def search_text(self, text):
        """Search text in the catalog."""



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
        arch = resnet50
        n = len(self.images_pkl)
        n_val = 27455
        val_idxs = list(range(n - n_val, n))

        tfms = tfms_from_model(arch, 224, transforms_side_on, max_zoom=1.1)
        md = ImageClassifierData.from_names_and_array(self.fake_path,
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

        self.learn = ConvLearner(md, models)  # , precompute=True)
        self.learn.opt_fn = partial(optim.Adam, betas=(0.9, 0.99))

        self.learn.load(os.path.join(self.model_path, 'pre0'))