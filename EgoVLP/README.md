# Task Graph Step Embeddings

## What is EgoVLP?

**EgoVLP** (Egocentric Video-Language Pretraining) is a vision-language model trained on egocentric videos from Ego4D. It learns a shared embedding space where **video features** and **text features** are aligned, enabling cross-modal matching between what is seen in a video and textual descriptions.

The model consists of:
- **Video encoder**: SpaceTimeTransformer (ViT-based) for visual features
- **Text encoder**: DistilBERT + projection layer for text features
- Both encoders project to a shared 256-dim space

## What we did here

We encoded the **textual step descriptions** from each CaptainCook4D task graph using the EgoVLP text encoder:

1. Loaded the pretrained EgoVLP checkpoint (`pretrained/egovlp.pth`)
2. For each task graph JSON, extracted only the actual steps (excluded START/END nodes)
3. Tokenized each step description with DistilBERT tokenizer
4. Encoded through DistilBERT + projection layer to get 256-dim embeddings

## Output format

Each `.pt` file contains:
```python
{
    'step_ids': [1, 2, 3, ...],           # Original node IDs from task graph
    'step_descriptions': ['...', ...],    # Text descriptions
    'embeddings': tensor(N, 256)          # EgoVLP text embeddings
}
```

These embeddings are now aligned with EgoVLP video features, enabling **step matching** via cosine similarity or Hungarian matching between detected video segments and task graph nodes.
