import csv
import numpy as np
import os
import glob
from collections import defaultdict

# -----------------------------
# Configuration for two runs
# -----------------------------
CONFIGS = [
    {
        "name": "256",
        "input_dir": "features_256",
        "output_dir": "step_embeddings_256_ActionFormer",
        "csv_file": "inferenza_finale_256.csv"
    },
    {
        "name": "768",
        "input_dir": "features_768",
        "output_dir": "step_embeddings_768_ActionFormer",
        "csv_file": "inferenza_finale_768.csv"
    }
]

# -----------------------------
# Feature Stride
# -----------------------------
FEATURE_STRIDE = 1.0 / 1.875

# -----------------------------
# Process each configuration
# -----------------------------
for config in CONFIGS:
    print(f"\n{'#' * 80}")
    print(f"# PROCESSING CONFIGURATION: {config['name']}")
    print(f"# Input Dir: {config['input_dir']}")
    print(f"# CSV File: {config['csv_file']}")
    print(f"# Output Dir: {config['output_dir']}")
    print(f"{'#' * 80}\n")

    INPUT_DIR = config["input_dir"]
    OUTPUT_DIR = config["output_dir"]
    CSV_FILE = config["csv_file"]

    # -----------------------------
    # Load CSV and group by video_id
    # -----------------------------
    print(f"Loading CSV file: {CSV_FILE}")
    step_predictions = defaultdict(list)

    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            t_start = float(row["t_start"])
            t_end = float(row["t_end"])
            score = float(row["score"])

            step_predictions[video_id].append({
                "t_start": t_start,
                "t_end": t_end,
                "score": score
            })

    print(f"Loaded predictions for {len(step_predictions)} videos")

    # Sort steps by start time for each video
    for video_id in step_predictions:
        step_predictions[video_id].sort(key=lambda x: x["t_start"])

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
        continue

    print(f"Found {len(input_files)} files to process in {INPUT_DIR}")

    processed_count = 0
    skipped_count = 0

    for file_path in input_files:
        filename = os.path.basename(file_path)
        # Extract recording ID from filename (e.g., "1_25" from "1_25_360p_224_0s_1s.npz")
        parts = filename.replace(".npz", "").split("_")
        RECORDING_ID = f"{parts[0]}_{parts[1]}"

        # Check if recording exists in CSV predictions
        if RECORDING_ID not in step_predictions:
            print(f"Warning: Recording ID {RECORDING_ID} not found in {CSV_FILE}, skipping...")
            skipped_count += 1
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing: {filename}")
        print(f"Recording ID: {RECORDING_ID}")
        print(f"{'=' * 80}")

        # -----------------------------
        # Load features
        # -----------------------------
        features_npz = np.load(file_path)
        features = features_npz["arr_0"]
        T, D = features.shape

        print(f"Features shape: {features.shape}")
        print(f"Feature stride: {FEATURE_STRIDE:.4f} sec")

        # -----------------------------
        # Get predicted steps for this recording
        # -----------------------------
        steps = step_predictions[RECORDING_ID]

        # -----------------------------
        # Extract step embeddings
        # -----------------------------
        step_embeddings = []

        print("\nSTEP EMBEDDINGS (CSV PREDICTIONS)")
        print("-" * 40)

        for i, step in enumerate(steps):
            start_t = step["t_start"]
            end_t = step["t_end"]
            score = step["score"]

            start_idx = int(start_t / FEATURE_STRIDE)
            end_idx = int(end_t / FEATURE_STRIDE)

            start_idx = max(0, min(start_idx, T - 1))
            end_idx = max(0, min(end_idx, T - 1))

            if end_idx <= start_idx:
                print(f"Warning: Step {i+1} skipped (empty interval)")
                continue

            step_feats = features[start_idx:end_idx + 1]
            step_embedding = step_feats.mean(axis=0)  # mean pooling

            step_embeddings.append({
                "recording_id": RECORDING_ID,
                "step_index": i,  # index in the CSV predictions
                "start_time": start_t,
                "end_time": end_t,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "score": score,
                "embedding": step_embedding
            })

            print(f"  Step {i+1}: {start_t:.2f}s -> {end_t:.2f}s | score: {score:.4f} | {end_idx - start_idx} frames")

        if not step_embeddings:
            print(f"Warning: No valid step embeddings extracted for {filename}, skipping...")
            skipped_count += 1
            continue

        # -----------------------------
        # Save step embeddings to NPZ
        # -----------------------------
        # Output filename: original name with "steps" before .npz
        output_filename = filename.replace(".npz", "_steps.npz")
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        embeddings = np.stack([s["embedding"] for s in step_embeddings])
        step_indices = np.array([s["step_index"] for s in step_embeddings], dtype=np.int32)
        scores = np.array([s["score"] for s in step_embeddings], dtype=np.float32)
        start_times = np.array([s["start_time"] for s in step_embeddings], dtype=np.float32)
        end_times = np.array([s["end_time"] for s in step_embeddings], dtype=np.float32)

        np.savez(
            output_path,
            embeddings=embeddings,
            step_indices=step_indices,
            scores=scores,
            start_times=start_times,
            end_times=end_times,
        )

        print(f"\nSaved: {output_path}")
        print(f"  embeddings shape: {embeddings.shape}")
        processed_count += 1

    print(f"\n{'=' * 80}")
    print(f"Configuration {config['name']} completed!")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"{'=' * 80}")

print(f"\n{'#' * 80}")
print("# ALL CONFIGURATIONS COMPLETED!")
print(f"{'#' * 80}")
