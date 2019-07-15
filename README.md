# Deep Search

Search engine that finds segments of videos based on the input text.

## How it works

Let's say you have a library of videos and you want to search for specific parts of the videos. Deep Search takes your library of videos and indexes them. Indexing will take some time. But after it is created and stored in the disc, searching is pretty fast. You can simply search by text. 

## Install the dependencies
The following steps are tested on Ubuntu 18.04.2 LTS.

0. Get Deep Search from git:
```
git clone https://github.com/arstepanyan/deep-search.git
cd deep-search
```
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
3. Install requirements for the core of Deep Search:
```
$ conda activate fastai
$ pip install -r requirements_core.txt
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
python deep-search.py search --catalog_path path-to-catalog --text text1 text2 textN
```