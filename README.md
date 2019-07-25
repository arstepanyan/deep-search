# Deep Search

Search service that finds segments of videos based on the input text.

## How it works

### Motivation  

Let's say you have a library of videos and you want to search for specific parts of the videos. Deep Search takes your library of videos and indexes them. Indexing will take some time. But after it is created and stored in the disc, searching is pretty fast. You can simply search by text. 

### Where does deep learning come into play?  

I tackle this problem by training a Deep Neural Network with ResNet50 as a backbone and 300 elements long dense layer instead of the last softmax layer of ResNet50. Why 300 element long Dense layer? Because given an input image, I want to extract features/embeddings from it that are as close to the 300 element word vector of the corresponding label as possible. I use around 1.5M Imagenet images of 1000 classes. The learning itself is done by challenging the model to minimize the cosine similarity loss between

1. The 300 elements dense layer (instead of the last softmax layer)
2. The 300 element word vector of the corresponding image's label from fastText

I trained the model using fastai library that works with Pytorch as its backend. On 2 GPUs (Titan XP, 12 GB), each epoch takes around 3 minutes to train. And after several experiments, I found that the model trained for 5 epochs does "pretty well". Pretty well here means having 0.076 cosine similarity loss on test images that haven't been seen during the training time. Here, cosine similarity loss is defined as (1 - cosine_similarity). Hence, 0 cosine similarity means perfect similarity.

Once the model is trained, a new/unlabeled video/photo library can be indexed by first extracting frames from videos then passing all the frames and photos through the trained model. That will result in embeddings for every frame and photo. It is indexed and saved on the disc. Finally, when an input text is given for the search of the short video clips, the fastText word vectors of that text's individual words are averaged and its nearest neighbors are found in the index. Then, based on the found nearest vectors (which correspond to frames or photos) short video clips are reconstructed and saved on disc.

But you don't have to worry about how to do all of the above steps. Just follow the below steps and your video library will be indexed and made available for searching of video clips.
 

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