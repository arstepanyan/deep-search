# Deep Search

Search engine that finds segments of videos based on the input image or text.

To check-out the repo:

    git clone git@github.com:arstepanyan/pythia.git
    
## How it works

Let's say you have a library of videos and you want to search for specific parts of the videos. Deep search takes your library of videos and indexes them. Indexing will take some time. But after it is created and stored in the dist, searching is pretty fast. You can give an image and ask to search for the parts of videos similar to that image. Or you can simply search by text. 

## Run The Notebook

To reproduce the results
1. Download the following videos from Youtube and store them in the directory *videos* (see the structure of the directories below)
    * Cactáceas, especies en peligro de extinción en SLP: https://www.youtube.com/watch?v=G1uoMICnZAk
    * Greek Layered Dip Recipe: https://www.youtube.com/watch?v=BRssfhRDLxo
    * Swiftcurrent Pass and Lookout - August 2012: https://www.youtube.com/watch?v=VUaQWSvn7Ik
2. Have a directory called *data* (with the structure shown below) in the same path as the *notebook* directory where the *image_video_search.ipynb* resides
       
       data/
       |
       |___features_mappings/
       |
       |___frames/
       |
       |___photots/
       |     |_______hills_river.jpg
       |
       |
       |___results_clips
       |
       |___videos/
             |_______Cactaceas_especies_en_peligro_de_extincion_en_SLP.mp4
             |_______Greek_Layered_Dip_Recipe.mp4
             |_______Swiftcurrent_Pass_and_Lookout_August_2012.mp4
             
3. Run the *notebook notebooks/image_video_search.ipynb*