import cv2
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.decomposition import PCA

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

if __name__ == "__main__":
    food = 'cereals'
    cams = ['webcam01']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    for cam in cams:
        name = f'P03_{cam}_P03_{food}'
        video_path = f'/data/benny_cai/video/dataset/P03/{cam}/P03_{food}.avi'
        frames = load_video_frames(video_path)
        frames = frames[:-2] # discard the last 2 frames to match with TSA
        print("video length: ", len(frames))

        preprocessed_frames = [preprocess(Image.fromarray(frame)).unsqueeze(0) for frame in frames] 
        preprocessed_frames = torch.cat(preprocessed_frames).to(device)
        print("preprocessed_frames.shape: ", preprocessed_frames.shape)
    
        with torch.no_grad():
            features = model.encode_image(preprocessed_frames)

        # Convert features to numpy array
        features_array = features.cpu().numpy()
        print("features.shape: ", features_array.shape)
        
        # Perform PCA to reduce dimensions
        pca = PCA(n_components=64)
        reduced_features = pca.fit_transform(features_array)
        print("reduced_features.shape: ", reduced_features.shape)  
    
        # save features
        path = f'/data/benny_cai/TSA-ActionSeg/datasets/CLIP_64/{name}.txt'
        np.savetxt(path, reduced_features)