import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm

# --------------------------------------------------------
# 1. SETUP PATHS & IMPORTS (Reuse your existing setup)
# --------------------------------------------------------
sys.path.append(os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'EgoVLP')):
    sys.path.append(os.path.join(os.getcwd(), 'EgoVLP'))

try:
    from model.model import FrozenInTime
except ImportError:
    print(" Error: Could not import FrozenInTime from EgoVLP/model/model.py")
    sys.exit(1)

# --------------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------------
# Input folder containing your (N, 768) .npz files
INPUT_FOLDER = "/kaggle/working/features_768"

# Output folder for (N, 256) files 
# (Can be the same if you want to overwrite, but safer to create a new one first)
OUTPUT_FOLDER = "/kaggle/working/features_256"

# Path to your weights
CHECKPOINT_PATH = "/kaggle/working/Features_extraction/feature_extractors/egovlp.pth"

# --------------------------------------------------------
# 3. HELPER: LOAD MODEL (Only need the projection layer)
# --------------------------------------------------------
def get_projection_layer(device):
    print(f" Loading EgoVLP weights from {CHECKPOINT_PATH}...")
    
    # Initialize model structure
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

    # Load weights
    if not os.path.exists(CHECKPOINT_PATH):
        print(" CRITICAL: Checkpoint not found!")
        sys.exit(1)

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if 'model' in checkpoint: state_dict = checkpoint['model']
    
    # Clean keys
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    # We ONLY need the video projection layer
    # In model.py: self.vid_proj = nn.Sequential(nn.Linear(ftr_dim, projection_dim))
    projection_layer = model.vid_proj
    projection_layer.to(device)
    projection_layer.eval()
    
    print(" Projection layer loaded successfully!")
    return projection_layer

# --------------------------------------------------------
# 4. MAIN CONVERSION LOOP
# --------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the projection head
    vid_proj = get_projection_layer(device)
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all .npz files
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.npz"))
    print(f"found {len(files)} feature files to convert.")

    for file_path in tqdm(files):
        try:
            # 1. Load Data
            with np.load(file_path) as data:
                # Usually saved as 'arr_0' by default in save_z
                feat_key = list(data.keys())[0]
                features_768 = data[feat_key] 
            
            # Check shape
            # Expected: (N, 768) or (1, N, 768)
            # If it's already 256, skip
            if features_768.shape[-1] == 256:
                continue

            # 2. Convert to Tensor
            input_tensor = torch.from_numpy(features_768).float().to(device)
            
            # 3. Apply Projection
            with torch.no_grad():
                features_256 = vid_proj(input_tensor)
            
            # 4. Save
            features_np = features_256.cpu().numpy()
            
            # Save with same filename in output folder
            filename = os.path.basename(file_path)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            
            np.savez(output_path, features_np)
            
        except Exception as e:
            print(f"Failed to convert {file_path}: {e}")

    print(" Conversion complete! Check folder:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()