import torch
import numpy as np
from core.topological_utils import compute_node_levels

def load_and_preprocess_data(path):
    # 1. Load the checkpoint
    print(f"Loading raw dataset from {path}...")
    checkpoint = torch.load(path, weights_only=False)
    raw_dataset = checkpoint['dataset']
    
    final_dataset = []
    groups = []

    print("Computing topological levels for all graphs...")
    for data in raw_dataset:
        num_nodes = data.x.size(0)
        
        # Forward levels: for standard root-to-leaf flow
        data.topo_level_fwd = compute_node_levels(data.edge_index, num_nodes)
        
        # Backward levels: for inverted leaf-to-root flow
        # We flip the edge_index [src, dst] -> [dst, src]
        edge_index_bwd = data.edge_index.flip(0) 
        data.topo_level_bwd = compute_node_levels(edge_index_bwd, num_nodes)
        
        final_dataset.append(data)
        
        # Collect groups for LOGO (Recipe ID ensures one recipe isn't in both train and val)
        groups.append(data.recipe_id)

    groups = np.array(groups)
    print(f"Processed {len(final_dataset)} graphs successfully.")
    
    return final_dataset, groups