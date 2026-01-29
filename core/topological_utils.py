import torch
def compute_node_levels(edge_index, num_nodes):
    src, dst = edge_index
    
    in_deg = torch.bincount(dst, minlength=num_nodes)
    
    # Inizialize level to -1
    node_levels = torch.full((num_nodes,), -1, dtype=torch.long)
    
    current_level = 0
    # Nodes with in-degree 0 are livel 0
    current_batch = (in_deg == 0).nonzero().view(-1)
    
    while current_batch.numel() > 0:
        
        node_levels[current_batch] = current_level
        
        # Remove "visited nodes"
        in_deg[current_batch] = -1 
        
        # Find next nodes
        mask = torch.isin(src, current_batch)
        affected_dst = dst[mask]
        
        if affected_dst.numel() > 0:
            in_deg[affected_dst] -= 1
        
        current_level += 1
        current_batch = (in_deg == 0).nonzero().view(-1)
        
    node_levels[node_levels == -1] = 0
    return node_levels