# Substep 3: Task-Graph Encoding and Step Matching

This document describes the implementation of Substep 3 from our project. The goal of this step is to create graph realizations by matching visual step embeddings from recipe videos to their corresponding nodes and text embeddings in the task graph.

## Quick Reference

| Script | Matching Method | Output Directory | Use Case |
|--------|----------------|------------------|----------|
| `task_graph_matching_individual.py` | Ground Truth (Oracle) | `graph_realizations_2/` | Best performance, requires annotations |
| `task_graph_matching_hungarian.py` | Hungarian Algorithm | `graph_realizations_hungarian/` | no annotations needed |
| `prepare_gnn_input.py` | - | `gnn_input/` | Prepare data for GNN (works with either) |


### Input Data

| Source | Format | Description |
|--------|--------|-------------|
| `step_embeddings/` | `.npz` files | Visual step embeddings extracted using EgoVLP encoder and then localized using actionformer or step_embeddings annotations fot best performance |
| `text_embeddings/` | `.pt` files | Text embeddings for task graph node descriptions, encoded using the same EgoVLP textual encoder |
| `task_graphs/` | `.json` files | Task graph structure with node descriptions and directed edges representing step dependencies |
| `recipe_mapping.csv` | CSV | Mapping from recipe IDs to recipe names |

#### Hungarian Matching (`task_graph_matching_individual_hungarian.py`)

This approach implements the **Hungarian algorithm** for optimal bipartite matching, as described in Substep 3 of the project specification:

1. Computes a **similarity matrix** between all visual step embeddings and all task graph node embeddings using cosine similarity
2. Applies the **Hungarian algorithm** to find the optimal one-to-one matching that maximizes total similarity
3. Matches each visual step to at most one task graph node and vice versa

**Cons:**
- Performance really depends on embedding quality, it may produce incorrect matches, especially for similar-looking steps

**Key Difference:**
- **Ground Truth**: Uses step IDs from `step_annotations.json` as oracle → perfect matching
- **Hungarian**: Uses cosine similarity → matching based on embedding similarity

### Hungarian Algorithm Details

The Hungarian matching implementation follows this pipeline:

1. **Similarity Matrix Construction:**
   ```
   S[i,j] = cosine_similarity(visual_embedding[i], text_embedding[j])
   ```

2. **Cost Matrix:**
   Since the Hungarian algorithm minimizes cost, we use:
   ```
   C[i,j] = -S[i,j]
   ```

3. **Optimal Assignment:**
   The algorithm finds the assignment that maximizes total similarity:
   ```
   argmax_π Σ S[i, π(i)]
   ```
   subject to the constraint that each visual step matches at most one node.

4. **Optional Threshold Filtering:**
   Matches below a similarity threshold can be filtered out to avoid poor matches.

The implementation uses `scipy.optimize.linear_sum_assignment`

### Feature Fusion

For each matched node, we combine the text embedding with the visual embedding using a weighted average:

```
node_feature = (1 - w) * text_embedding + w * visual_embedding
```

where `w = 0.5` by default. This simple fusion strategy can be replaced with a learnable projection module during training.

For **unmatched nodes** (steps that were skipped in the video), the original text embedding is preserved without modification.

### Video-Level Labels

The video-level label for task verification is computed from the step-level error annotations:
- **Correct (0)**: No matched step contains an error
- **Incorrect (1)**: At least one matched step contains an error

## Output

### Graph Realizations

The output is saved to individual `.pt` files in either:
- `graph_realizations_2/` (ground truth matching)
- `graph_realizations_hungarian/` (Hungarian matching)

Each file `{recipe_id}_{recording_id}.pt` contains a dictionary with the following structure:

```python
{
    'node_features': torch.Tensor,      # (num_nodes, 256) - fused node embeddings
    'node_ids': list,                   # Local step IDs for each node
    'node_descriptions': list,          # Text descriptions for each node
    'edges': list,                      # Directed edges [(src, dst), ...]
    'steps': dict,                      # Full step descriptions from task graph
    'match_info': dict,                 # Matching details (timing, errors, similarity)
    'num_matched': int,                 # Number of successfully matched nodes
    'num_visual_steps': int,            # Number of visual steps in the video
    'num_graph_nodes': int,             # Number of nodes in the task graph
    'recipe_id': int,
    'recording_id': int,
    'recipe_name': str
}
```

### GNN-Ready Dataset

A processed dataset suitable for GNN training is saved to `gnn_input/gnn_dataset.pt`:

```python
{
    'dataset': list,    # List of Data objects with x, edge_index, y, match_mask
    'recipes': dict,    # Mapping from recipe names to recording IDs
    'num_correct': int,
    'num_incorrect': int
}
```

### File Structure

```
gnn_input/
└── gnn_dataset.pt          # Combined dataset for GNN training
```

### Single Graph Attributes

Each graph in `dataset` has the following attributes:

| Attribute | Shape/Type | Description |
|-----------|------------|-------------|
| `x` | `(num_nodes, 256)` | Node features (combined text+visual) |
| `edge_index` | `(2, num_edges)` | Graph edges in COO format |
| `y` | `(1,)` tensor | Label: 0=correct, 1=incorrect |
| `match_mask` | `(num_nodes,)` bool | Which nodes were matched to visual steps |
| `num_nodes` | int | Number of nodes in the graph |
| `num_matched` | int | Number of matched nodes |
| `num_visual_steps` | int | Number of visual steps in the video |
| `recipe_id` | int | Recipe identifier |
| `recording_id` | int | Recording identifier |
| `recipe_name` | str | Recipe name (for leave-one-out CV) |

