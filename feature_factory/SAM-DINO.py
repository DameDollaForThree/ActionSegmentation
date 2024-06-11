import numpy as np
import torch
from PIL import Image
from semantic_sam import (
    SemanticSamAutomaticMaskGenerator,
    build_model,
    build_semantic_sam,
    plot_results,
    prepare_image,
)
from semantic_sam.BaseModel import BaseModel
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageSegmentation
import yaml
from sklearn.cluster import DBSCAN
import functools
import operator
import cv2
from sklearn.decomposition import PCA
import time

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

def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split(".")
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(
                pointer, dict
            ), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v

def load_opt_from_config_file(conf_file):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files: config file path

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    with open(conf_file, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    load_config_dict_to_opt(opt, config_dict)

    return opt

# Class definition for ObjectEncoder
class ObjectEncoder:
    def __init__(
        self,
        dino_model="facebook/dinov2-large",
        semantic_sam_configs="/data/benny_cai/video/semantic_SAM/semantic_sam_only_sa-1b_swinL.yaml",
        semantic_sam_ckpt="/data/benny_cai/video/semantic_SAM/swinl_only_sam_many2many.pth",
        device="cuda:7",
        dbscan_eps=30,
        dbscan_min_samples=3,
        segment_area_threshold=0.0003,
        mask_num_threshold=5,
    ) -> None:
        # Initialize DINO model processor and model
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model)
        self.dino_model = AutoModel.from_pretrained(dino_model)
        self.dino_model.to(device)

        # Set device and extract CUDA ID
        self.device = device
        assert device.startswith("cuda:")
        self.cuda_id = int(device.split(":")[1])
        print("CUDA: ", self.cuda_id)

        # Initialize Semantic-SAM model with specified configuration
        with torch.cuda.device(self.cuda_id):
            opt = load_opt_from_config_file(semantic_sam_configs)
            self.sematic_sam = (
                BaseModel(opt, build_model(opt))
                .from_pretrained(semantic_sam_ckpt)
                .eval()
                .cuda()
            )
            self.mask_generator = SemanticSamAutomaticMaskGenerator(self.sematic_sam)

        # Set thresholds and DBSCAN parameters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.segment_area_threshold = segment_area_threshold
        self.mask_num_threshold = mask_num_threshold

    # Encode an image by segmenting, filtering, and embedding
    def encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)

        # Segment the image using Semantic-SAM model
        segments = self._segment_image(image)
        segments_masks = [segment["segmentation"] for segment in segments]

        # Encode the image using the DINO model
        patch_embeddings = self._dino_encode_image(image) # (37, 49, 1024)
        
        # Merge similar segments using DBSCAN clustering
        merged_segments = self._merge_similar_segments(segments_masks, patch_embeddings)
        
        # Get non-overlapping segments by eliminating overlaps
        non_overlapping_segments = self._get_non_overlapping_segments(merged_segments)

        # Get embeddings for non-overlapping segments
        filtered_masks, embeddings = self._get_mask_embedding(non_overlapping_segments, patch_embeddings)
        print("number of filtered masks: ", len(filtered_masks))
        
        # average all embeddings
        avg_embedding = np.mean(np.array(embeddings), axis=0)

        return filtered_masks, avg_embedding

    # Segment an image using Semantic-SAM model
    def _segment_image(self, image):
        with torch.cuda.device(self.cuda_id):
            _, input_image = self._prepare_semantic_sam_image(image) 
            masks = self.mask_generator.generate(input_image)
        return masks

    # Prepare image for Semantic-SAM model by resizing and converting to tensor
    def _prepare_semantic_sam_image(self, image):
        t = []
        t.append(transforms.Resize(640, interpolation=Image.BICUBIC))
        transform1 = transforms.Compose(t)
        image_ori = transform1(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
        return image_ori, images

    # Encode an image using the DINO model
    def _dino_encode_image(self, image):
        inputs = self.dino_processor(
            images=image,
            return_tensors="pt",
            size={"shortest_edge": self.dino_model.config.image_size},
            do_center_crop=False,
        )
        with torch.inference_mode():
            outputs = self.dino_model(
                pixel_values=inputs["pixel_values"].to(self.dino_model.device)
            )
        patches = outputs.last_hidden_state[0, 1:, ...].cpu().numpy()

        h, _ = inputs["pixel_values"].shape[-2:]
        patch_h = h // self.dino_model.config.patch_size
        patches = patches.reshape(patch_h, -1, patches.shape[-1])

        return patches

    # Get embeddings for given masks using patch embeddings
    def _get_mask_embedding(self, masks, patch_embeddings):
        h, w, _ = patch_embeddings.shape
        embeddings = []
        idx_remove = []
        for i, mask in enumerate(masks):
            resized_mask = transforms.functional.resize(
                torch.from_numpy(mask[None, ...]).to(torch.float), (h, w)
            )
            if resized_mask.sum() < self.mask_num_threshold:
                idx_remove.append(i)
                continue
            resized_mask = resized_mask.squeeze(0).numpy()
            embedding = patch_embeddings * resized_mask[..., None]
            embedding = embedding.sum(axis=(0, 1)) / resized_mask.sum()
            embeddings.append(embedding)

        masks = [item for idx, item in enumerate(masks) if idx not in idx_remove]
        return masks, embeddings

    # Merge similar segments using DBSCAN clustering
    def _merge_similar_segments(self, masks, patch_embeddings):
        _, embeddings = self._get_mask_embedding(masks, patch_embeddings)
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        cluster_assignment = dbscan.fit_predict(embeddings)

        merged_segments = [mask for mask, a in zip(masks, cluster_assignment) if a < 0]
        for i in range(cluster_assignment.max() + 1):
            similar_masks = [mask for mask, a in zip(masks, cluster_assignment) if a == i]
            merged_mask = functools.reduce(operator.or_, similar_masks)
            merged_segments.append(merged_mask)

        return merged_segments

    # Get non-overlapping segments by eliminating overlaps
    def _get_non_overlapping_segments(self, masks):
        union = np.array(False)
        output = []
        for mask in sorted(masks, key=np.sum):
            unique = mask & (~union)
            if unique.sum() / unique.size < self.segment_area_threshold:
                continue
            union = union | mask
            output.append(unique)
        return output

if __name__ == "__main__":
    encoder = ObjectEncoder()
    food = 'cereals'
    cam = 'webcam01'
    name = f'P03_{cam}_P03_{food}'
    video_path = f'/data/benny_cai/video/dataset/P03/{cam}/P03_{food}.avi'
    frames = load_video_frames(video_path)
    frames = frames[:-2] # discard the last 2 frames
    print("video length: ", len(frames))
    
    ts = time.time()
    features = []
    for i, frame in enumerate(frames):
        print("image ", i)
        image = Image.fromarray(frame)
        _, avg_embedding = encoder.encode_image(image)
        assert avg_embedding.shape[0] == 1024
        features.append(avg_embedding)
    print("took: ", time.time() - ts, " seconds")
    
    features = np.array(features)
    print("features.shape: ", features.shape)
    
    # Perform PCA to reduce dimensions
    pca1 = PCA(n_components=64)
    pca1_reduced_features = pca1.fit_transform(features) 
    path = f'/data/benny_cai/TSA-ActionSeg/datasets/SAM_DINO_avg_64/{name}.txt'
    np.savetxt(path, pca1_reduced_features)
