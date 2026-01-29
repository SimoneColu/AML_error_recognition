import sys
import torchvision
import os
import torch.multiprocessing as mp 

# ==========================================
# 1. FIX COMPATIBILITY (TORCHVISION)
# ==========================================
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional

# ==========================================
# 2.  SETUP EGOVLP PATHS
# ==========================================
sys.path.append(os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'EgoVLP')):
    sys.path.append(os.path.join(os.getcwd(), 'EgoVLP'))

# ==========================================
# 3. IMPORTS
# ==========================================
import argparse
import datetime
import glob
import logging
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from natsort import natsorted

# PytorchVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

# Torchvision
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo

# EgoVLP Model Import
try:
    from model.model import FrozenInTime
except ImportError:
    print(" Warning: Could not import FrozenInTime. Check EgoVLP path.")

# Mock parse_config for EgoVLP
from types import ModuleType
if 'parse_config' not in sys.modules:
    mock_module = ModuleType("parse_config")
    class MockConfigParser:
        def __init__(self, *args, **kwargs): pass
    mock_module.ConfigParser = MockConfigParser
    sys.modules["parse_config"] = mock_module    

# ==========================================
# 4.  ARGUMENTS & LOGGING
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="egovlp", help="Specify the method to be used.")
    parser.add_argument(
        "--prefixes", 
        nargs="+", 
        default=["1_", "2_"], 
        help="List of prefixes to filter videos (e.g. 1_ 2_)"
    )
    args, _ = parser.parse_known_args()
    return args

# Logging setup (safe for multiprocessing)
def setup_logger(rank):
    logging.basicConfig(
        filename=f"logs/extraction_gpu_{rank}.log", 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(f"Worker-{rank}")

# ==========================================
# 5.  CORE CLASSES & FUNCTIONS
# ==========================================

class VideoProcessor:
    def __init__(self, method, feature_extractor, video_transform, device, logger):
        self.method = method
        self.feature_extractor = feature_extractor
        self.video_transform = video_transform
        self.device = device
        self.logger = logger

    def process_video(self, video_name, video_directory_path, output_features_path):
        # Frequency setup (ActionFormer: 1.875 Hz)
        segment_size = 1.0 / 1.875 
        
        if not video_name.endswith(".mp4"):
            video_filename = f"{video_name}.mp4"
        else:
            video_filename = video_name
            
        video_path = os.path.join(video_directory_path, video_filename)
        base_name = video_name.replace(".mp4", "")
        output_file_path = os.path.join(output_features_path, base_name)
        expected_filename = f"{output_file_path}_{int(segment_size)}s_{int(1)}s.npz"

        # Skip if exists
        if os.path.exists(expected_filename) and os.path.getsize(expected_filename) > 100:
            return

        try:
            video = EncodedVideo.from_path(video_path)
            video_duration = video.duration
        except Exception as e:
            self.logger.error(f"Error opening video {video_name}: {e}")
            return

        segment_end = max(video_duration - segment_size + 0.001, 0.0)
        video_features = []
        
        for start_time in np.arange(0, segment_end, segment_size):
            end_time = start_time + segment_size
            end_time = min(end_time, video_duration)

            if end_time - start_time < 0.1: 
                continue

            try:
                video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
                segment_video_inputs = video_data["video"]

                segment_features = extract_features(
                    video_data_raw=segment_video_inputs,
                    feature_extractor=self.feature_extractor,
                    transforms_to_apply=self.video_transform,
                    method=self.method,
                    device=self.device # <--- Passed explicitly
                )
                video_features.append(segment_features)
            except Exception as e:
                self.logger.error(f"Error extracting segment {start_time} on {video_name}: {e}")
                continue

        if len(video_features) > 0:
            video_features = np.vstack(video_features)
            np.savez(expected_filename, video_features)
        else:
            self.logger.warning(f"No valid features for {video_name}")

def extract_features(video_data_raw, feature_extractor, transforms_to_apply, method, device):
    # Prepare transformation
    video_data_for_transform = {"video": video_data_raw, "audio": None}
    video_data = transforms_to_apply(video_data_for_transform)
    video_inputs = video_data["video"]

    # Prepare Tensor (Batch dimension)
    if method == "egovlp":
        # Move to specific GPU immediately
        video_input = video_inputs.unsqueeze(0).to(device) 
        # (Batch, Channels, Time, H, W) -> (Batch, Time, Channels, H, W)
        video_input = video_input.permute(0, 2, 1, 3, 4)
    else:
        video_input = video_inputs.unsqueeze(0).to(device)

    with torch.no_grad():
        if method == "egovlp":
            features = feature_extractor.video_model(video_input)
            if hasattr(feature_extractor, 'video_proj'):
                features = feature_extractor.video_proj(features)
        else:
            features = feature_extractor(video_input)
            
        if isinstance(features, tuple):
            features = features[0]
            
    return features.cpu().numpy()

def get_video_transformation(name):
    if name == "egovlp":
        num_frames = 16
        side_size = 256
        crop_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        video_transform = Compose([
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size),
        ])
        return ApplyTransformToKey(key="video", transform=video_transform)
    return None

def get_feature_extractor(name, device):
    if name == "egovlp":
        print(f"[{device}]  Loading EgoVLP architecture...")
        
        model = FrozenInTime(
            video_params={
                "model": "SpaceTimeTransformer",
                "arch_config": "base_patch16_224",
                "num_frames": 16,
                "pretrained": False, 
                "time_init": "zeros"
            },
            text_params={
                "model": "bert-base-uncased",
                "pretrained": True,
                "input": "text"
            },
            projection_dim=256,
            load_checkpoint=None
        )

        checkpoint_path = "/kaggle/working/Features_extraction/feature_extractors/egovlp.pth"
        
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            if 'model' in checkpoint: state_dict = checkpoint['model']
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            model.load_state_dict(new_state_dict, strict=False)
            print(f"[{device}]  Weights loaded!")
        else:
            print(f"[{device}]  CRITICAL ERROR: Weights not found at {checkpoint_path}")
            
    else:
        model = torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinB_epic")
        model.heads = torch.nn.Identity()

    # Move model to the specific GPU assigned to this worker
    model = model.to(device)
    model = model.eval()
    return model

# ==========================================
# 6.  PARALLEL WORKER LOGIC
# ==========================================

def process_on_gpu(gpu_id, video_list, args, video_files_path, output_features_path):
    """
    Worker function that runs on a specific GPU.
    """
    device = f"cuda:{gpu_id}"
    print(f" Worker started on {device} processing {len(video_list)} videos.")
    
    # Setup specific logger for this process
    worker_logger = setup_logger(gpu_id)
    
    # Initialize Model & Transform LOCALLY on this GPU
    # (Important: Model must be loaded inside the process to be on the right GPU RAM)
    try:
        feature_extractor = get_feature_extractor(args.backbone, device=device)
        video_transform = get_video_transformation(args.backbone)
        
        processor = VideoProcessor(args.backbone, feature_extractor, video_transform, device, worker_logger)
        
        # Create output dir if not exists (thread safe enough usually, but good to check)
        os.makedirs(output_features_path, exist_ok=True)

        for video_file in tqdm(video_list, desc=f"GPU {gpu_id}", position=gpu_id):
            processor.process_video(video_file, video_files_path, output_features_path)
            
    except Exception as e:
        print(f" Critical Error on {device}: {e}")
        worker_logger.error(f"Critical Error: {e}")

# ==========================================
# 7. 🎬 MAIN
# ==========================================
def main():
    # Force 'spawn' for CUDA multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Paths
    video_files_path = "/kaggle/working/videos_unzip"
    base_output_path = "/kaggle/working/Features_extraction"
    os.makedirs("logs", exist_ok=True)

    args = parse_arguments()
    output_features_path = os.path.join(base_output_path, f"result_{args.backbone}")

    print(f" Input Videos: {video_files_path}")
    print(f" Output: {output_features_path}")

    if not os.path.exists(video_files_path):
        print(f" Error: Folder {video_files_path} not found.")
        return

    # Filter Files
    all_files = [f for f in os.listdir(video_files_path) if f.endswith(".mp4")]
    target_prefixes = tuple(args.prefixes)
    mp4_files = [f for f in all_files if f.startswith(target_prefixes)]
    mp4_files = natsorted(mp4_files)

    print(f" Total filtered videos: {len(mp4_files)}")

    if len(mp4_files) == 0:
        print(" No videos found.")
        return

    # === GPU ALLOCATION ===
    num_gpus = torch.cuda.device_count()
    print(f" Detected {num_gpus} GPUs.")

    if num_gpus < 2:
        print(" Warning: Less than 2 GPUs detected. Running sequentially on GPU 0.")
        process_on_gpu(0, mp4_files, args, video_files_path, output_features_path)
    else:
        # Split work into 2 chunks
        mid_point = len(mp4_files) // 2
        files_gpu_0 = mp4_files[:mid_point]
        files_gpu_1 = mp4_files[mid_point:]

        print(f" GPU 0 assigned: {len(files_gpu_0)} videos")
        print(f" GPU 1 assigned: {len(files_gpu_1)} videos")

        # Launch Processes
        p1 = mp.Process(target=process_on_gpu, args=(0, files_gpu_0, args, video_files_path, output_features_path))
        p2 = mp.Process(target=process_on_gpu, args=(1, files_gpu_1, args, video_files_path, output_features_path))

        p1.start()
        p2.start()
        
        p1.join()
        p2.join()
        print(" All processes finished.")

if __name__ == "__main__":
    main()