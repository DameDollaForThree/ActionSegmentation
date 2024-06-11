import numpy as np

foods = ["cereals"]
cams = ["webcam01"]

for food in foods:
    for cam in cams:
        # CLIP
        CLIP_feat_file = f"/data/benny_cai/TSA-ActionSeg/datasets/CLIP_64/P03_{cam}_P03_{food}.txt"
        CLIP_feat = []

        with open(CLIP_feat_file) as f:
            lines = f.readlines()

        for feature in lines:
            CLIP_feat.append(np.array(feature.split(' ')).astype(float))
    
        CLIP_feat = np.array(CLIP_feat)
        print(CLIP_feat.shape)

        # SAM+DINO
        SAM_DINO_feat_file = f"/data/benny_cai/TSA-ActionSeg/datasets/SAM_DINO_avg_64/P03_{cam}_P03_{food}.txt"
        SAM_DINO_feat = []

        with open(SAM_DINO_feat_file) as f:
            lines = f.readlines()

        for feature in lines:
            SAM_DINO_feat.append(np.array(feature.split(' ')).astype(float))
    
        SAM_DINO_feat = np.array(SAM_DINO_feat)
        print(SAM_DINO_feat.shape)

        feat = np.concatenate((CLIP_feat, SAM_DINO_feat), axis=1)
        print(feat.shape)
        
        name = "CLIP_64+SAM_DINO_avg_64"
        path = f'/data/benny_cai/TSA-ActionSeg/datasets/{name}/P03_{cam}_P03_{food}.txt'
        np.savetxt(path, feat)
