# Deep Search

Search engine that finds segments of videos based on the input text.

To check-out the repo:

    git clone git@github.com:arstepanyan/deep-search.git
    
## How it works

Let's say you have a library of videos and you want to search for specific parts of the videos. Deep search takes your library of videos and indexes them. Indexing will take some time. But after it is created and stored in the disc, searching is pretty fast. You can simply search by text. 

## Install the dependencies
1. Get fastai from source:
 ```
$ git clone https://github.com/fastai/fastai.git
$ cd fastai
```
2. Setup conda environment for GPU:
```
$ conda env create -f environment.yml
$ conda activate fastai
```
and for CPU:
```
$ conda env create -f environment-cpu.yml
$ conda activate fastai-cpu
```
Delete the fastai repo:
```
$ cd ..; rm -rf fastai
```
3. Downgrade fastai and install dependencies:
```
$ pip install matplotlib=3.0.0
$ pip install fastai==0.7.0
$ pip install torchtext==0.2.3
$ conda install -c akode nmslib
$ conda install -c conda-forge time
$ conda install -c conda-forge opencv
$ pip install opencv-python
$ pip install -U fasttext
$ pip install moviepy
$ pip install pytube
```

## Running the app

First, activate fastai conda environment: 
```
$ conda activate fastai
```
To index your catalog
```
python deep-search.py index --catalog_path path-to-catalog
```
To search in your catalog
```
python deep-search.py search --catalog_path path-to-catalog --results_path path-to-results --text text1 text2 textN
```