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
        
        # Caricamento corretto del .npy
        self.data = np.load(features_path, allow_pickle=True)

        with open('annotations/annotation_json/error_annotations.json', 'r') as f:
            self._error_annotations = json.load(f)

        # Crea la mappa {recording_id: is_error} (o label numerica)
        # Nota: Assicuriamoci che gli ID coincidano con quelli del filename
        self.video_id_error_map = {
            item['recording_id']: (1.0 if item['is_error'] else 0.0) 
            for item in self._error_annotations
        }

        # Caso 1: array scalare che contiene un dict o una list
        if isinstance(self.data, np.ndarray) and self.data.shape == ():
            self.data = self.data.item()

        # Caso 2: dict {video_id: sample}
        if isinstance(self.data, dict):
            self.data = list(self.data.values())

        # Controllo di sicurezza
        if not isinstance(self.data, (list, tuple)):
            raise TypeError(
                f"Formato non supportato: {type(self.data)}. "
                "Atteso list o dict di sample."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Restituisce un singolo campione.
        Output:
            features: Tensor (Num_Steps, Feature_Dim)
            label: Float (0.0 o 1.0)
            video_id: str (utile per debug)
        """
        sample = self.data[idx]
        
        # Recupera le feature (sequenza di step)
        # Assumiamo siano salvate come tensori o numpy array
        features = torch.tensor(sample['features'], dtype=torch.float32)
        
        # Opzionale: ritorniamo anche l'ID per tracciare quale ricetta stiamo analizzando
        video_id = sample.get('recording_id', str(idx))

        # --- DEBUG ---
        if video_id not in self.video_id_error_map:
            print(f"ERRORE CRITICO ALL'INDICE {idx}:")
            print(f"  -> ID cercato: '{video_id}'")
            print(f"  -> Esempi di chiavi valide nel dizionario: {list(self.video_id_error_map.keys())[:5]}")
            # Se vuoi evitare il crash immediato per vedere gli altri errori, decommenta:
            # return features, torch.tensor([0.0]), video_id 
        # -------------

        # Recupera la label (0 = Corretto, 1 = Errato o viceversa in base alla tua convenzione)
        raw_label = self.video_id_error_map[video_id] 
        label = torch.tensor(raw_label, dtype=torch.float32)

        return features, label, video_id

def recipe_collate_fn(batch):
    """
    Funzione per gestire batch con ricette di lunghezza diversa.
    Aggiunge padding (zeri) alle sequenze più corte.
    
    Returns:
        padded_features: (Batch, Max_Len, Feature_Dim)
        labels: (Batch,)
        masks: (Batch, Max_Len) -> True se è padding, False se è dato reale
        ids: list of video_ids
    """
    # batch è una lista di tuple restituite da __getitem__
    features, labels, ids = zip(*batch)
    
    # Troviamo la lunghezza massima nel batch corrente
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    feature_dim = features[0].shape[1]
    
    batch_size = len(features)
    
    # Inizializziamo i tensori di padding (tutti zeri)
    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    # Maschera: True indica che la posizione è padding (da ignorare)
    # Nota: In PyTorch Transformer spesso True=Ignore. Verifica sempre la documentazione.
    # Qui usiamo: True = Padding (da ignorare).
    masks = torch.ones(batch_size, max_len, dtype=torch.bool) 
    
    for i, seq in enumerate(features):
        end = lengths[i]
        padded_features[i, :end, :] = seq
        masks[i, :end] = False # Le posizioni con dati reali sono False (non ignorare)
        
    labels = torch.stack(labels)
    
    return padded_features, labels, masks, ids