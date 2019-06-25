# Deep Search

Search engine that finds segments of videos based on the input image or text.

## Pipeline

To reproduce the results
1. Download the following videos from Youtube
    * (Cactaceas_especies_en_peligro_de_extincion_en_SLP)[]
    * (Greek_Layered_Dip_Recipe)[]
    * (Swiftcurrent_Pass_and_Lookout_August_2012)[]
2. Have a directory called *data* (with the structure shown bellow) in the same path as the *notebook* directory where the *image_video_search.ipynb* resides
       
       data/
       |
       |___features_mappings/
       |
       |___frames/
       |
       |___photots/
       |
       |___results_clips
       |
       |___videos/
             |__Cactaceas_especies_en_peligro_de_extincion_en_SLP.mp4
             |__Greek_Layered_Dip_Recipe.mp4
             |__Swiftcurrent_Pass_and_Lookout_August_2012.mp4
             
3. Run the *notebook notebooks/image_video_search.ipynb*