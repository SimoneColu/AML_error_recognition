import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import json

class CaptainCookRecipeDataset(Dataset):
    def __init__(self, features_path):
        """
        Dataset per il task di Task Verification (Estensione).
        Carica le feature pre-calcolate a livello di ricetta.
        
        Args:
            features_path (str): Percorso al file .npy creato nel Substep 1
                                 contenente dizionario {videoid_start_end: "features"}
        """
        super().__init__()
        
        # Loading .npy
        self.data = np.load(features_path, allow_pickle=True)

        with open('annotations/annotation_json/error_annotations.json', 'r') as f:
            self._error_annotations = json.load(f)

        # Create the map {video_id: is_error} 
        self.video_id_error_map = {
            item['recording_id']: (1.0 if item['is_error'] else 0.0) 
            for item in self._error_annotations
        }

        # Case 1: Scalar array (0-d) containing a list/dict
        if isinstance(self.data, np.ndarray) and self.data.shape == ():
            self.data = self.data.item()

        # Case 2: 1D Numpy Array containing objects 
        elif isinstance(self.data, np.ndarray):
            self.data = self.data.tolist()

        # Case 3: Dict {video_id: sample} -> Convert to list
        if isinstance(self.data, dict):
            self.data = list(self.data.values())

        # Safety Check
        if not isinstance(self.data, (list, tuple)):
            raise TypeError(
                f"Unsupported format: {type(self.data)}. "
                "Atteso list o dict di samle."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Retrieve features
        features = torch.tensor(sample['features'], dtype=torch.float32)
        
        video_id = sample.get('video_id', str(idx))

        # --- DEBUG ---
        if video_id not in self.video_id_error_map:
            print(f"ERRORE CRITICO ALL'INDICE {idx}:")
            print(f"  -> ID cercato: '{video_id}'")
            print(f"  -> Esempi di chiavi valide nel dizionario: {list(self.video_id_error_map.keys())[:5]}")
            # return features, torch.tensor([0.0]), video_id 
        # -------------

        # Get the label (0 = Corect, 1 = Error)
        raw_label = self.video_id_error_map[video_id] 
        label = torch.tensor(raw_label, dtype=torch.float32)

        return features, label, video_id

def recipe_collate_fn(batch):

    features, labels, ids = zip(*batch)
    
    # max length of current batch
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    feature_dim = features[0].shape[1]
    
    batch_size = len(features)
    
    # initialize padding tensor (all zeros)
    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    # Mask: True means position is padding (to ignore)
    masks = torch.ones(batch_size, max_len, dtype=torch.bool) 
    
    for i, seq in enumerate(features):
        end = lengths[i]
        padded_features[i, :end, :] = seq
        masks[i, :end] = False 
        
    labels = torch.stack(labels)
    
    return padded_features, labels, masks, ids