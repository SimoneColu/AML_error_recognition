import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool, AttentionalAggregation
from torch_geometric.data import Data
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax

class DAGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Attention components: determines how much to trust each parent node
        self.attn_w1 = nn.Linear(hidden_dim, 1, bias=False) 
        self.attn_w2 = nn.Linear(hidden_dim, 1, bias=False)

        # Gated update: combines aggregated neighbor info with the node's own previous state
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, node_levels):
        src, dst = edge_index
        h_cur = x.clone()      # dynamic state: updated as we traverse levels
        h_prev_layer = x       # static state: fixed features from the previous laye
        
        max_level = node_levels.max().item()
        
        # Level-by-level iteration to respect the DAG's topological dependency
        for level in range(max_level + 1):
            batch_nodes_mask = (node_levels == level)
            if not batch_nodes_mask.any():
                continue
            
            # Retriving node indexes from non-zero value inside the collection
            batch_nodes_idx = batch_nodes_mask.nonzero().view(-1)
            
            # Identify edges pointing to nodes in the current level
            edge_mask = batch_nodes_mask[dst]
            
            if edge_mask.any():
                # Select only src and dst nodes in the current level
                rel_src = src[edge_mask] 
                rel_dst = dst[edge_mask]
                
                # Query comes from the "old" state, Key comes from "new" parent states
                query = self.attn_w1(h_prev_layer[rel_dst])  
                key = self.attn_w2(h_cur[rel_src]) 
                
                # Compute attention scores and normalize across incoming edges for each node
                scores = query + key
                alpha = softmax(scores, rel_dst, num_nodes=x.size(0))
                
                # Message passing: weight the parent features by attention scores
                weighted_messages = h_cur[rel_src] * alpha
                
                # Sum messages for the target nodes (dst)
                msgs_batch = scatter_sum(weighted_messages, rel_dst, dim=0, dim_size=x.size(0))[batch_nodes_idx]
            else:
                # Leaf nodes or nodes with no parents in this graph
                msgs_batch = torch.zeros((batch_nodes_idx.size(0), self.hidden_dim), device=x.device)

            # GRU update: h_batch_new = GRU(old_state, aggregated_parent_messages)
            h_batch_prev = h_prev_layer[batch_nodes_idx]
            h_batch_new = self.gru(h_batch_prev, msgs_batch)
            
            # Update h_cur so the NEXT level can use these refined features
            h_cur[batch_nodes_idx] = h_batch_new

        return h_cur

class RecipeVerifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.norms_fwd = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms_bwd = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.fwd_layers = nn.ModuleList([DAGNNLayer(hidden_dim) for _ in range(num_layers)])
        self.bwd_layers = nn.ModuleList([DAGNNLayer(hidden_dim) for _ in range(num_layers)])

        cat_dim = (num_layers + 1) * hidden_dim
        readout_dim = 4 * cat_dim
        
        self.att_pool_fwd = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(cat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        self.att_pool_bwd = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(cat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )

        self.final_norm = nn.LayerNorm(readout_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        N = x.size(0)
        num_graphs = batch.max().item() + 1

        topo_fwd = data.topo_level_fwd
        topo_bwd = data.topo_level_bwd

        # Create the inverted graph for the backward pass (reversing edge direction)
        edge_index_bwd = torch.stack([edge_index[1], edge_index[0]], dim=0)
        
        # Initial projection: Map raw features (text+vision) to the hidden dimension
        # Resulting h0 is the "Level 0" representation (h^0) 
        h0 = F.relu(self.input_proj(x))

        # FORWARD PASS: Root-to-Leaf Information Flow
        # Skip Connections (h_new + h_fwd) and LayerNorm to prevent 
        # vanishing gradients and over-smoothing across the DAG levels.
        h_fwd = h0
        outputs_fwd = [h0]
        for i, layer in enumerate(self.fwd_layers):
            h_new = layer(h_fwd, edge_index, topo_fwd)
            h_fwd = self.norms_fwd[i](h_fwd + h_new)
            outputs_fwd.append(h_fwd)

        # BACKWARD PASS: Leaf-to-Root Information Flow
        # This treats the 'goal' of the recipe as the starting point, 
        # sending requirements 'up' the graph to the initial ingredients.
        h_bwd = h0
        outputs_bwd = [h0]
        for i, layer in enumerate(self.bwd_layers):
            h_new = layer(h_bwd, edge_index_bwd, topo_bwd)
            h_bwd = self.norms_bwd[i](h_bwd + h_new)
            outputs_bwd.append(h_bwd)

        # For each node, we concatenate the representations from ALL layers.
        # This allows the model to "see" local info (h0), local structure (h1), 
        # and global graph position (h2) simultaneously.
        H_fwd_cat = torch.cat(outputs_fwd, dim=1)    # Shape: [N, (num_layers + 1) * hidden_dim]
        H_bwd_cat = torch.cat(outputs_bwd, dim=1)

        # TOPOLOGICAL POOLING (Readout)
        # Instead of pooling all nodes, focus on the endpoints:
        # - For Forward: Target nodes (Leaves, out-degree=0) represent the final result.
        # - For Backward: Source nodes (Roots, in-degree=0) represent the foundation.
        out_degree = torch.bincount(edge_index[0], minlength=N)
        target_mask = (out_degree == 0)

        in_degree = torch.bincount(edge_index[1], minlength=N)
        source_mask = (in_degree == 0)
        
        fwd_max = global_max_pool(H_fwd_cat[target_mask], batch[target_mask], size=num_graphs)
        fwd_att = self.att_pool_fwd(H_fwd_cat[target_mask], batch[target_mask])

        bwd_max = global_max_pool(H_bwd_cat[source_mask], batch[source_mask], size=num_graphs)
        bwd_att = self.att_pool_bwd(H_bwd_cat[source_mask], batch[source_mask])

        # Final Graph Representation: Concatenates forward/backward and max/att views
        graph_emb = torch.cat([fwd_max, fwd_att, bwd_max, bwd_att], dim=1)
        graph_emb = self.final_norm(graph_emb)

        logits = self.classifier(graph_emb)

        return logits