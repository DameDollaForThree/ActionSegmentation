# Unsupervised Temporal Action Segmentation from Video
This project proposes some extension works based on the [TSA (temporal-semantic aware)](https://arxiv.org/abs/2304.06403) paper.

### Core idea:
- Combine high-level semantic feature with low-level region features to obtain better frame-wise representations.
- Perform TSA feature transform that accounts for temporal and semantic similarity between frames. 
- Unsupervised clustering without labels.

### Repo Structure:
- `feature_factory`: source code to extract CLIP, SAM+DINO, CLIP+SAM+DINO features, and text embedding from videos
- `TSA`: source code for the TSA pipeline

### Dataset:
- [The Breakfast Actions Dataset](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/)
- Pre-computed IDT features for the dataset can be downloaded from [here](https://github.com/Annusha/unsup_temp_embed/blob/master/HOWTO_master.md)

### Pipeline:
1. Feature extraction of the input video
    - CLIP / SAM+DINO / CLIP+SAM+DINO / CLIP+Text
2. TSA feature transform
3. Feature Clustering 
    - FINCH, Kmeans, Spectral, TW-FINCH

### Requirements:
- [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM), [CLIP](https://github.com/openai/CLIP), [TSA](https://github.com/elenabbbuenob/TSA-ActionSeg)
- Note: the python environment for feature extraction and TSA can be different since they are two separate modules.




