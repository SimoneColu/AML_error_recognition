"""
Encode task graph steps using EgoVLP text encoder
This script encodes only the actual steps (excluding START/END) from each task graph
output: .pt file per task graph containing step embeddings
"""
import os
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class EgoVLPTextEncoder(nn.Module):
    """Simplified EgoVLP text encoder (DistilBERT + projection layer)"""

    def __init__(self, projection_dim=256):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(
            'distilbert-base-uncased',
            cache_dir='pretrained/distilbert-base-uncased'
        )
        # projection layer to align with video embedding space
        self.txt_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.text_model.config.hidden_size, projection_dim),
        )

    def forward(self, text_data):
        """Encode text and project to shared embedding space."""
        text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def load_from_egovlp_checkpoint(self, checkpoint_path):
        """Load text encoder weights from full EgoVLP checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']

        # extract only text-related weights
        text_model_dict = {}
        txt_proj_dict = {}

        for key, value in state_dict.items():
            if key.startswith('text_model.'):
                new_key = key.replace('text_model.', '')
                text_model_dict[new_key] = value
            elif key.startswith('txt_proj.'):
                new_key = key.replace('txt_proj.', '')
                txt_proj_dict[new_key] = value

        if text_model_dict:
            self.text_model.load_state_dict(text_model_dict, strict=False)
            print(f"Loaded text_model weights ({len(text_model_dict)} keys)")

        if txt_proj_dict:
            self.txt_proj.load_state_dict(txt_proj_dict, strict=True)
            print(f"Loaded txt_proj weights ({len(txt_proj_dict)} keys)")


def load_task_graph(json_path):
    """Load task graph and extract only actual steps (exclude START/END)"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    steps = data['steps']
    filtered_steps = {}

    for step_id, description in steps.items():
        # Skip START and END nodes
        if description.upper() in ['START', 'END']:
            continue
        filtered_steps[int(step_id)] = description

    return filtered_steps


def encode_task_graph(model, tokenizer, steps, device, batch_size=32):
    """
    Encode all steps in a task graph.
    returns:
        dict with:
            - 'step_ids': list of step IDs (int)
            - 'step_descriptions': list of text descriptions
            - 'embeddings': tensor of shape (num_steps, embedding_dim)
    """
    step_ids = sorted(steps.keys())
    descriptions = [steps[sid] for sid in step_ids]

    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(descriptions), batch_size):
            batch_texts = descriptions[i:i+batch_size]

            # Tokenize
            tokens = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=77
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            # Encode
            embeddings = model(tokens)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)

    return {
        'step_ids': step_ids,
        'step_descriptions': descriptions,
        'embeddings': all_embeddings
    }


def main():
    parser = argparse.ArgumentParser(description='Encode task graph steps with EgoVLP text encoder')
    parser.add_argument('--task_graphs_dir', type=str, default='task_graphs',
                        help='Directory containing task graph JSON files')
    parser.add_argument('--output_dir', type=str, default='task_graph_embeddings',
                        help='Directory to save encoded embeddings')
    parser.add_argument('--checkpoint', type=str, default='pretrained/egovlp.pth',
                        help='Path to EgoVLP checkpoint (optional)')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    print(f"Initializing EgoVLP text encoder (projection_dim={args.projection_dim})...")
    model = EgoVLPTextEncoder(projection_dim=args.projection_dim)

    # Load checkpoint if available
    if os.path.exists(args.checkpoint):
        print(f"Loading EgoVLP checkpoint from {args.checkpoint}...")
        model.load_from_egovlp_checkpoint(args.checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Using DistilBERT with randomly initialized projection layer.")
        print("For proper EgoVLP embeddings, download the checkpoint from the EgoVLP repo.")

    model = model.to(args.device)
    model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Get all task graph files
    task_graph_files = [f for f in os.listdir(args.task_graphs_dir) if f.endswith('.json')]
    print(f"\nFound {len(task_graph_files)} task graph files")

    # Process each task graph
    for filename in sorted(task_graph_files):
        json_path = os.path.join(args.task_graphs_dir, filename)
        task_name = os.path.splitext(filename)[0]

        print(f"\nProcessing: {task_name}")

        # Load and filter steps
        steps = load_task_graph(json_path)
        print(f"  Found {len(steps)} steps (excluding START/END)")

        # Encode steps
        encoded = encode_task_graph(model, tokenizer, steps, args.device)

        # Save embeddings
        output_path = os.path.join(args.output_dir, f"{task_name}.pt")
        torch.save(encoded, output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Embedding shape: {encoded['embeddings'].shape}")

    print(f"\nDone! Encoded {len(task_graph_files)} task graphs.")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
