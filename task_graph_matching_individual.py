"""
Task-Graph encoding + Step matching 
This script creates one graph realization file per recording
Each file is saved as {recipe_id}_{recording_id}.pt
Usage:
    python task_graph_matching_individual.py
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
import csv

# ============================================================
# configuration
# ============================================================
STEP_EMBEDDINGS_DIR = "step_embeddings"
TEXT_EMBEDDINGS_DIR = "text_embeddings"
TASK_GRAPHS_DIR = "task_graphs"
OUTPUT_DIR = "graph_realizations_2"
RECIPE_MAPPING_FILE = "recipe_mapping.csv"

# ============================================================
# helper Functions
# ============================================================

def load_recipe_mapping(mapping_file):
    """Load recipe_id to recipe_name mapping from CSV"""
    mapping = {}
    with open(mapping_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            recipe_id = int(row['activity_idx'])
            recipe_name = row['activity_name'].lower().replace(" ", "")
            mapping[recipe_id] = recipe_name
    return mapping

def parse_step_embedding_filename(filename):
    """Extract recipe_id and recording_id from filename like '10_16_360p_224_0s_1s_steps.npz'"""
    parts = filename.replace('.npz', '').split('_')
    recipe_id = int(parts[0])
    recording_id = int(parts[1])
    return recipe_id, recording_id

def load_step_embeddings(filepath):
    """Load visual step embeddings from npz file"""
    data = np.load(filepath)
    return {
        'embeddings': data['embeddings'],
        'step_ids': data['step_ids'],
        'has_errors': data['has_errors'],
        'start_times': data['start_times'],
        'end_times': data['end_times']
    }

def load_text_embeddings(filepath):
    """Load text embeddings for task graph nodes"""
    data = torch.load(filepath, weights_only=False)
    return {
        'step_ids': data['step_ids'],
        'step_descriptions': data['step_descriptions'],
        'embeddings': data['embeddings'].numpy()
    }

def load_task_graph(filepath):
    """Load task graph from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_similarity(visual_emb, text_emb):
    """compute cosine similarity between two embeddings"""
    visual_norm = visual_emb / (np.linalg.norm(visual_emb) + 1e-8)
    text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
    return np.dot(visual_norm, text_norm)

def convert_global_to_local_step_ids(step_ids, text_node_ids):
    """Convert global step IDs to local step IDs if needed"""
    step_ids = np.array(step_ids)
    min_text_id = min(text_node_ids)
    max_text_id = max(text_node_ids)

    min_step_id = step_ids.min()
    max_step_id = step_ids.max()

    if min_step_id >= min_text_id and max_step_id <= max_text_id:
        return step_ids

    offset = min_step_id - min_text_id
    local_ids = step_ids - offset
    return local_ids

def create_graph_realization(task_graph, text_embeddings, visual_embeddings,
                              visual_step_data, projection_weight=0.5):
    """create a graph realization using GROUND TRUTH step_ids mapping"""
    num_text_nodes = len(text_embeddings['step_ids'])
    embedding_dim = text_embeddings['embeddings'].shape[1]

    node_features = text_embeddings['embeddings'].copy()
    text_node_id_to_idx = {nid: idx for idx, nid in enumerate(text_embeddings['step_ids'])}

    local_step_ids = convert_global_to_local_step_ids(
        visual_step_data['step_ids'],
        text_embeddings['step_ids']
    )

    match_info = {}
    matched_text_indices = set()

    for vis_idx, gt_node_id in enumerate(local_step_ids):
        gt_node_id = int(gt_node_id)

        if gt_node_id in text_node_id_to_idx:
            text_idx = text_node_id_to_idx[gt_node_id]

            combined = (1 - projection_weight) * node_features[text_idx] + \
                       projection_weight * visual_embeddings[vis_idx]
            node_features[text_idx] = combined
            matched_text_indices.add(text_idx)

            similarity = compute_similarity(
                visual_embeddings[vis_idx],
                text_embeddings['embeddings'][text_idx]
            )

            match_info[gt_node_id] = {
                'visual_step_idx': int(vis_idx),
                'visual_step_id': gt_node_id,
                'has_error': bool(visual_step_data['has_errors'][vis_idx]),
                'start_time': float(visual_step_data['start_times'][vis_idx]),
                'end_time': float(visual_step_data['end_times'][vis_idx]),
                'similarity': float(similarity)
            }

    realization = {
        'node_features': node_features,
        'node_ids': text_embeddings['step_ids'],
        'node_descriptions': text_embeddings['step_descriptions'],
        'edges': task_graph['edges'],
        'steps': task_graph['steps'],
        'match_info': match_info,
        'num_matched': len(match_info),
        'num_visual_steps': len(visual_embeddings),
        'num_graph_nodes': num_text_nodes,
        'unmatched_visual_indices': [],
        'unmatched_text_indices': [i for i in range(num_text_nodes) if i not in matched_text_indices]
    }

    return realization

def save_individual_realization(realization, output_path):
    """Save a single graph realization to a .pt file"""
    output_data = {
        'node_features': torch.tensor(realization['node_features'], dtype=torch.float32),
        'node_ids': realization['node_ids'],
        'node_descriptions': realization['node_descriptions'],
        'edges': realization['edges'],
        'steps': realization['steps'],
        'match_info': realization['match_info'],
        'num_matched': realization['num_matched'],
        'num_visual_steps': realization['num_visual_steps'],
        'num_graph_nodes': realization['num_graph_nodes'],
        'recipe_id': realization['recipe_id'],
        'recording_id': realization['recording_id'],
        'recipe_name': realization['recipe_name']
    }
    torch.save(output_data, output_path)

# ============================================================
# main 
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading recipe mapping...")
    recipe_mapping = load_recipe_mapping(RECIPE_MAPPING_FILE)
    print(f"  Found {len(recipe_mapping)} recipes")

    step_files = sorted([f for f in os.listdir(STEP_EMBEDDINGS_DIR) if f.endswith('.npz')])
    print(f"\nFound {len(step_files)} step embedding files")

    summary_data = []

    for i, step_file in enumerate(step_files):
        recipe_id, recording_id = parse_step_embedding_filename(step_file)

        if recipe_id not in recipe_mapping:
            print(f"  Warning: Recipe ID {recipe_id} not found in mapping, skipping {step_file}")
            continue

        recipe_name = recipe_mapping[recipe_id]
        
        text_emb_path = os.path.join(TEXT_EMBEDDINGS_DIR, f"{recipe_name}.pt")
        if not os.path.exists(text_emb_path):
            print(f"  Warning: Text embeddings not found for {recipe_name}, skipping")
            continue

        task_graph_path = os.path.join(TASK_GRAPHS_DIR, f"{recipe_name}.json")
        if not os.path.exists(task_graph_path):
            print(f"  Warning: Task graph not found for {recipe_name}, skipping")
            continue

        step_emb_path = os.path.join(STEP_EMBEDDINGS_DIR, step_file)
        visual_data = load_step_embeddings(step_emb_path)
        text_data = load_text_embeddings(text_emb_path)
        task_graph = load_task_graph(task_graph_path)

        realization = create_graph_realization(
            task_graph=task_graph,
            text_embeddings=text_data,
            visual_embeddings=visual_data['embeddings'],
            visual_step_data=visual_data,
            projection_weight=0.5
        )

        recording_key = f"{recipe_id}_{recording_id}"
        realization['recipe_id'] = recipe_id
        realization['recording_id'] = recording_id
        realization['recipe_name'] = recipe_name

        # save individual file
        output_path = os.path.join(OUTPUT_DIR, f"{recording_key}.pt")
        save_individual_realization(realization, output_path)

        # collect summary data
        avg_sim = np.mean([info['similarity'] for info in realization['match_info'].values()]) \
                  if realization['match_info'] else 0
        has_errors = any(info['has_error'] for info in realization['match_info'].values())

        summary_data.append({
            'recording_id': recording_key,
            'recipe_id': recipe_id,
            'recipe_name': recipe_name,
            'num_visual_steps': realization['num_visual_steps'],
            'num_graph_nodes': realization['num_graph_nodes'],
            'num_matched': realization['num_matched'],
            'avg_similarity': avg_sim,
            'has_errors': has_errors
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(step_files)} recordings...")

    print(f"\nTotal realizations created: {len(summary_data)}")
    print(f"Output directory: {OUTPUT_DIR}/")

    # save summary CSV
    summary_path = os.path.join(OUTPUT_DIR, "realizations_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'recording_id', 'recipe_id', 'recipe_name', 'num_visual_steps',
            'num_graph_nodes', 'num_matched', 'avg_similarity', 'has_errors'
        ])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    main()
