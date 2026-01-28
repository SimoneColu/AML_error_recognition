import json
import numpy as np
import os
import glob

# -----------------------------
# Paths
# -----------------------------
INPUT_DIR = "features_768"
OUTPUT_DIR = "step_embeddings_768"
STEPS_FILE = "step_annotations.json"

# -----------------------------
# Feature Stride
# -----------------------------
FEATURE_STRIDE = 1.0 / 1.875

# -----------------------------
# Load step annotations
# -----------------------------
with open(STEPS_FILE, "r") as f:
    step_data = json.load(f)

# -----------------------------
# Ensure output directory exists
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Process all files in input directory
# -----------------------------
input_files = glob.glob(os.path.join(INPUT_DIR, "*.npz"))

if not input_files:
    print(f"No .npz files found in {INPUT_DIR}")
    exit(1)

print(f"Found {len(input_files)} files to process in {INPUT_DIR}")

for file_path in input_files:
    filename = os.path.basename(file_path)
    # Extract recording ID from filename (e.g., "1_25" from "1_25_360p_224_0s_1s.npz")
    parts = filename.replace(".npz", "").split("_")
    RECORDING_ID = f"{parts[0]}_{parts[1]}"

    print(f"\n{'=' * 80}")
    print(f"Processing: {filename}")
    print(f"Recording ID: {RECORDING_ID}")
    print(f"{'=' * 80}")

    # Check if recording exists in step annotations
    if RECORDING_ID not in step_data:
        print(f"Warning: Recording ID {RECORDING_ID} not found in step_annotations.json, skipping...")
        continue

    # -----------------------------
    # Load features
    # -----------------------------
    features_npz = np.load(file_path)
    features = features_npz["arr_0"]
    T, D = features.shape

    print(f"Features shape: {features.shape}")
    print(f"Feature stride: {FEATURE_STRIDE:.4f} sec")

    # -----------------------------
    # Get steps for this recording
    # -----------------------------
    steps = step_data[RECORDING_ID]["steps"]

    # -----------------------------
    # Extract step embeddings
    # -----------------------------
    step_embeddings = []

    print("\nSTEP EMBEDDINGS (GROUND TRUTH)")
    print("-" * 40)

    for i, step in enumerate(steps):
        start_t = step["start_time"]
        end_t = step["end_time"]

        start_idx = int(start_t / FEATURE_STRIDE)
        end_idx = int(end_t / FEATURE_STRIDE)

        start_idx = max(0, min(start_idx, T - 1))
        end_idx = max(0, min(end_idx, T - 1))

        if end_idx <= start_idx:
            print(f"Warning: Step {step['step_id']} skipped (empty interval)")
            continue

        step_feats = features[start_idx:end_idx + 1]
        step_embedding = step_feats.mean(axis=0)  # mean pooling

        step_embeddings.append({
            "recording_id": RECORDING_ID,
            "step_id": step["step_id"],
            "start_time": start_t,
            "end_time": end_t,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "embedding": step_embedding,
            "has_errors": step["has_errors"],
            "description": step["description"]
        })

        print(f"  Step {i+1}: {step['step_id']} | {start_t:.2f}s -> {end_t:.2f}s | {end_idx - start_idx} frames")

    if not step_embeddings:
        print(f"Warning: No valid step embeddings extracted for {filename}, skipping...")
        continue

    # -----------------------------
    # Save step embeddings to NPZ
    # -----------------------------
    # Output filename: original name with "steps" before .npz
    output_filename = filename.replace(".npz", "_steps.npz")
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    embeddings = np.stack([s["embedding"] for s in step_embeddings])
    step_ids = np.array([s["step_id"] for s in step_embeddings], dtype=np.int32)
    has_errors = np.array([s["has_errors"] for s in step_embeddings], dtype=np.bool_)
    start_times = np.array([s["start_time"] for s in step_embeddings], dtype=np.float32)
    end_times = np.array([s["end_time"] for s in step_embeddings], dtype=np.float32)
    
    np.savez(
        output_path,
        embeddings=embeddings,
        step_ids=step_ids,
        has_errors=has_errors,
        start_times=start_times,
        end_times=end_times,
    )

    print(f"\nSaved: {output_path}")
    print(f"  embeddings shape: {embeddings.shape}")

print(f"\n{'=' * 80}")
print("Processing completed!")
print(f"{'=' * 80}")
