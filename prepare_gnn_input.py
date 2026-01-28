"""
Prepare Graph Realizations for Substep 4
this script:
1. loads graph realization files from Substep 3 (graph_realizations_2/)
2. creates video-level labels (correct=0, incorrect=1)
3. formats data for PyTorch Geometric GNN training
4. includes a learnable projection for combining text and video features
Usage:
    python prepare_gnn_input.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from glob import glob

# ============================================================
# Configuration
# ============================================================
REALIZATIONS_DIR = "graph_realizations_2"  # directory with individual .pt files
OUTPUT_DIR = "gnn_input"

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Note: torch_geometric not installed. Using compatible dict format.")

    # Simple Data class that mimics PyG Data
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __repr__(self):
            return f"Data({', '.join(f'{k}={type(v).__name__}' for k, v in self.__dict__.items())})"

# ============================================================
# Learnable Projection Module
# ============================================================

class FeatureProjection(nn.Module):
    """
    Learnable projection for combining text and visual features
    """
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=256):
        super().__init__()

        # Project text features
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Project visual features
        self.visual_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Combine features
        self.combine = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_emb, visual_emb=None, has_match=None):
        """
        Args:
            text_emb: (N, D) text embeddings for all nodes
            visual_emb: (N, D) visual embeddings (zeros for unmatched)
            has_match: (N,) boolean mask indicating matched nodes
        Returns:
            combined: (N, D) combined node features
        """
        text_proj = self.text_proj(text_emb)
        if visual_emb is None:
            return text_proj
        visual_proj = self.visual_proj(visual_emb)
        # concatenate and combine
        combined = torch.cat([text_proj, visual_proj], dim=-1)
        output = self.combine(combined)
        # for unmatched nodes, use only text projection
        if has_match is not None:
            output = torch.where(
                has_match.unsqueeze(-1),
                output,
                text_proj
            )
        return output

# ============================================================
# Data Preparation Functions
# ============================================================

def create_edge_index(edges, node_id_to_idx, num_nodes):
    """
    convert edge list to PyTorch Geometric edge_index format
    Args:
        edges: list of [src, dst] pairs with node IDs
        node_id_to_idx: dict mapping node IDs to indices
    Returns:
        edge_index: (2, num_edges) tensor
    """
    valid_edges = []
    for src, dst in edges:
        # skip edges involving START (0) or END (last node)
        if src in node_id_to_idx and dst in node_id_to_idx:
            valid_edges.append([node_id_to_idx[src], node_id_to_idx[dst]])

    if not valid_edges:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(valid_edges, dtype=torch.long).T
    return edge_index


def compute_video_label(match_info):
    """
    compute video-level label from step-level errors
    Returns:
        0 if all matched steps are correct (no errors)
        1 if any matched step has an error
    """
    if not match_info:
        return 0  # no matches, assume correct

    for info in match_info.values():
        if info['has_error']:
            return 1  # at least one error

    return 0  # all correct


def load_individual_realization(filepath):
    return torch.load(filepath, weights_only=False)


def create_pyg_data(realization):
    """
    create a PyTorch Geometric Data object from a graph realization.
    Args:
        realization: dict with realization data (loaded from individual .pt file)
    Returns:
        data object for GNN training
    Keys used from realization:
        - node_features: (num_nodes, 256) combined text+visual features
        - node_ids: list of step IDs
        - edges: list of [src, dst] pairs
        - match_info: dict with has_error flags for computing label
        - recipe_id, recording_id, recipe_name: metadata
    """
    # get node features (already combined in Substep 3)
    node_features = realization['node_features']

    if isinstance(node_features, np.ndarray):
        node_features = torch.tensor(node_features, dtype=torch.float32)

    num_nodes = node_features.shape[0]

    # create node ID to index mapping
    node_ids = realization['node_ids']
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # create edge index
    edge_index = create_edge_index(
        realization['edges'],
        node_id_to_idx,
        num_nodes
    )

    # compute video-level label from match_info has_error flags
    y = compute_video_label(realization['match_info'])

    # create match mask (which nodes were matched to visual steps)
    match_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for node_id in realization['match_info'].keys():
        if node_id in node_id_to_idx:
            match_mask[node_id_to_idx[node_id]] = True

    # create Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=torch.tensor([y], dtype=torch.long),
        match_mask=match_mask,
        num_nodes=num_nodes,
        num_matched=realization['num_matched'],
        num_visual_steps=realization['num_visual_steps'],
        recipe_id=realization['recipe_id'],
        recording_id=realization['recording_id'],
        recipe_name=realization['recipe_name']
    )

    return data


def prepare_dataset(realizations_dir, output_dir):
    """
    Prepare the full dataset for GNN training from individual .pt files
    """
    # find all individual realization files
    pt_files = sorted(glob(os.path.join(realizations_dir, "*.pt")))
    pt_files = [f for f in pt_files if not f.endswith("summary.pt")]
    print(f"Loading graph realizations from {realizations_dir}/")
    print(f"  Found {len(pt_files)} individual realization files")
    # process all realizations
    dataset = []
    labels = []

    for filepath in pt_files:
        # Load individual file
        realization = load_individual_realization(filepath)
        # Convert to PyG Data
        data = create_pyg_data(realization)
        dataset.append(data)
        labels.append(data.y.item())

    # statistics
    num_correct = labels.count(0)
    num_incorrect = labels.count(1)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Correct executions: {num_correct} ({100*num_correct/len(dataset):.1f}%)")
    print(f"  Incorrect executions: {num_incorrect} ({100*num_incorrect/len(dataset):.1f}%)")

    # group by recipe for leave-one-out evaluation
    recipes = {}
    for data in dataset:
        recipe_name = data.recipe_name
        if recipe_name not in recipes:
            recipes[recipe_name] = []
        recipes[recipe_name].append(data)

    print(f"\nRecipes: {len(recipes)}")
    for recipe_name, samples in sorted(recipes.items()):
        correct = sum(1 for d in samples if d.y.item() == 0)
        incorrect = len(samples) - correct
        print(f"  {recipe_name}: {len(samples)} samples ({correct} correct, {incorrect} incorrect)")

    # Save processed dataset
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "gnn_dataset.pt")
    torch.save({
        'dataset': dataset,
        'recipes': {k: [d.recording_id for d in v] for k, v in recipes.items()},
        'num_correct': num_correct,
        'num_incorrect': num_incorrect
    }, output_path)
    print(f"\nSaved dataset to: {output_path}")

    return dataset, recipes


def print_example_graph(data):
    """Print details of an example graph for verification."""
    print("\n" + "="*80)
    print("EXAMPLE GRAPH FOR GNN INPUT")
    print("="*80)
    print(f"Recipe: {data.recipe_name}")
    print(f"Recording ID: {data.recording_id}")
    print(f"Label: {'INCORRECT' if data.y.item() == 1 else 'CORRECT'}")
    print(f"\nGraph Structure:")
    print(f"  - Node features shape: {data.x.shape}")
    print(f"  - Edge index shape: {data.edge_index.shape}")
    print(f"  - Number of nodes: {data.num_nodes}")
    print(f"  - Number of edges: {data.edge_index.shape[1]}")
    print(f"  - Matched nodes: {data.match_mask.sum().item()}/{data.num_nodes}")
    print(f"\nNode features (first 5 nodes, first 10 dims):")
    print(data.x[:5, :10])
    print(f"\nEdge index (first 10 edges):")
    print(data.edge_index[:, :10])
    print("="*80)

# ============================================================
# Main
# ============================================================

def main():
    # prepare dataset from individual files
    dataset, recipes = prepare_dataset(REALIZATIONS_DIR, OUTPUT_DIR)
    if len(dataset) == 0:
        print("No data found. Make sure you have run task_graph_matching_individual.py first.")
        return

    print("\n" + "="*80)
    print("Data is ready for Substep 4")
    print("="*80)

if __name__ == "__main__":
    main()