"""
Task-Graph encoding + hungarian Matching
this script creates one graph realization file per recording using Hungarian algorithm
for optimal matching between visual step embeddings and task graph nodes
Each file is saved as {recipe_id}_{recording_id}.pt
Usage:
    python task_graph_matching_individual_hungarian.py
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import csv

# ============================================================
# conf
# ============================================================
STEP_EMBEDDINGS_DIR = "step_embeddings"
TEXT_EMBEDDINGS_DIR = "text_embeddings"
TASK_GRAPHS_DIR = "task_graphs"
OUTPUT_DIR = "graph_realizations_hungarian"
RECIPE_MAPPING_FILE = "recipe_mapping.csv"

# ============================================================
# helper functions
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
    """Compute cosine similarity between two embeddings"""
    visual_norm = visual_emb / (np.linalg.norm(visual_emb) + 1e-8)
    text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
    return np.dot(visual_norm, text_norm)

def compute_similarity_matrix(visual_embeddings, text_embeddings):
    """
    Compute pairwise cosine similarity matrix between visual and text embeddings.
    Args:
        visual_embeddings: (N, D) array of visual step embeddings
        text_embeddings: (M, D) array of text node embeddings
    Returns:
        similarity_matrix: (N, M) matrix where entry [i,j] is similarity
                            between visual step i and text node j
    """
    # normalize embeddings
    visual_norm = visual_embeddings / (np.linalg.norm(visual_embeddings, axis=1, keepdims=True) + 1e-8)
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
    # compute similarity matrix (N x M)
    similarity_matrix = np.dot(visual_norm, text_norm.T)
    return similarity_matrix

def hungarian_matching(similarity_matrix):
    """
    Apply Hungarian algorithm to find optimal one-to-one matching
    Args:
        similarity_matrix: (N, M) matrix of similarities
    Returns:
        visual_indices: array of matched visual step indices
        text_indices: array of matched text node indices
    """
    # hungarian algorithm minimizes cost, so we use negative similarity
    cost_matrix = -similarity_matrix
    # find optimal assignment
    visual_indices, text_indices = linear_sum_assignment(cost_matrix)
    return visual_indices, text_indices

def create_graph_realization_hungarian(task_graph, text_embeddings, visual_embeddings,
                                       visual_step_data, projection_weight=0.5,
                                       similarity_threshold=0.0):
    """
    Create a graph realization using hungarian matching algorithm
    Args:
        task_graph: task graph structure
        text_embeddings: text embeddings for task graph nodes
        visual_embeddings: visual embeddings from video steps
        projection_weight: weight for combining visual and text embeddings
        similarity_threshold: minimum similarity to consider a match valid (to remove some noise)
    Returns:
        realization: dictionary containing the graph realization
    """
    num_text_nodes = len(text_embeddings['step_ids'])
    num_visual_steps = len(visual_embeddings)
    embedding_dim = text_embeddings['embeddings'].shape[1]

    # initialize with text embeddings
    node_features = text_embeddings['embeddings'].copy()
    text_node_id_to_idx = {nid: idx for idx, nid in enumerate(text_embeddings['step_ids'])}

    # compute similarity matrix between all visual steps and all text nodes
    similarity_matrix = compute_similarity_matrix(visual_embeddings, text_embeddings['embeddings'])

    # apply Hungarian matching
    visual_indices, text_indices = hungarian_matching(similarity_matrix)

    # process matches
    match_info = {}
    matched_text_indices = set()
    unmatched_visual_indices = []

    for vis_idx, text_idx in zip(visual_indices, text_indices):
        similarity = similarity_matrix[vis_idx, text_idx]

        # only accept matches above threshold
        if similarity < similarity_threshold:
            unmatched_visual_indices.append(int(vis_idx))
            continue

        # get the text node ID
        text_node_id = text_embeddings['step_ids'][text_idx]

        # combine visual and text embeddings
        combined = (1 - projection_weight) * node_features[text_idx] + \
                   projection_weight * visual_embeddings[vis_idx]
        node_features[text_idx] = combined
        matched_text_indices.add(text_idx)

        # store match information
        match_info[text_node_id] = {
            'visual_step_idx': int(vis_idx),
            'text_node_idx': int(text_idx),
            'visual_step_id': text_node_id,  # assigned based on matching
            'has_error': bool(visual_step_data['has_errors'][vis_idx]),
            'start_time': float(visual_step_data['start_times'][vis_idx]),
            'end_time': float(visual_step_data['end_times'][vis_idx]),
            'similarity': float(similarity)
        }

    # find unmatched visual steps (those below threshold)
    all_matched_visual = set(visual_indices)
    for vis_idx in range(num_visual_steps):
        if vis_idx not in all_matched_visual:
            unmatched_visual_indices.append(vis_idx)

    realization = {
        'node_features': node_features,
        'node_ids': text_embeddings['step_ids'],
        'node_descriptions': text_embeddings['step_descriptions'],
        'edges': task_graph['edges'],
        'steps': task_graph['steps'],
        'match_info': match_info,
        'num_matched': len(match_info),
        'num_visual_steps': num_visual_steps,
        'num_graph_nodes': num_text_nodes,
        'unmatched_visual_indices': unmatched_visual_indices,
        'unmatched_text_indices': [i for i in range(num_text_nodes) if i not in matched_text_indices],
        'similarity_matrix': similarity_matrix.tolist()  
    }
    return realization


def save_individual_realization(realization, output_path):
    """Save a single graph realization to a .pt file"""
    # remove similarity matrix before saving 
    save_realization = {k: v for k, v in realization.items() if k != 'similarity_matrix'}

    output_data = {
        'node_features': torch.tensor(save_realization['node_features'], dtype=torch.float32),
        'node_ids': save_realization['node_ids'],
        'node_descriptions': save_realization['node_descriptions'],
        'edges': save_realization['edges'],
        'steps': save_realization['steps'],
        'match_info': save_realization['match_info'],
        'num_matched': save_realization['num_matched'],
        'num_visual_steps': save_realization['num_visual_steps'],
        'num_graph_nodes': save_realization['num_graph_nodes'],
        'unmatched_visual_indices': save_realization['unmatched_visual_indices'],
        'unmatched_text_indices': save_realization['unmatched_text_indices'],
        'recipe_id': save_realization['recipe_id'],
        'recording_id': save_realization['recording_id'],
        'recipe_name': save_realization['recipe_name']
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
    print("\nProcessing with Hungarian matching algorithm...")

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

        # load
        step_emb_path = os.path.join(STEP_EMBEDDINGS_DIR, step_file)
        visual_data = load_step_embeddings(step_emb_path)
        text_data = load_text_embeddings(text_emb_path)
        task_graph = load_task_graph(task_graph_path)

        # create realization using Hungarian matching
        realization = create_graph_realization_hungarian(
            task_graph=task_graph,
            text_embeddings=text_data,
            visual_embeddings=visual_data['embeddings'],
            visual_step_data=visual_data,
            projection_weight=0.5,
            similarity_threshold=0.0  # accept all hungarian matches
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
            'num_unmatched_visual': len(realization['unmatched_visual_indices']),
            'num_unmatched_text': len(realization['unmatched_text_indices']),
            'avg_similarity': avg_sim,
            'has_errors': has_errors
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(step_files)} recordings...")

    print(f"\nTotal realizations created: {len(summary_data)}")
    print(f"Output directory: {OUTPUT_DIR}/")

    # Save summary CSV
    summary_path = os.path.join(OUTPUT_DIR, "realizations_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'recording_id', 'recipe_id', 'recipe_name', 'num_visual_steps',
            'num_graph_nodes', 'num_matched', 'num_unmatched_visual',
            'num_unmatched_text', 'avg_similarity', 'has_errors'
        ])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"Saved summary to: {summary_path}")

    # Print statistics
    avg_similarity = np.mean([item['avg_similarity'] for item in summary_data])
    print(f"\nStatistics:")
    print(f"  Average matching similarity: {avg_similarity:.4f}")
    print(f"  Average matched steps per video: {np.mean([item['num_matched'] for item in summary_data]):.2f}")

if __name__ == "__main__":
    main()
