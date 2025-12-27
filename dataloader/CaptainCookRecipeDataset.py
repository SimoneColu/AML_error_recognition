import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class CaptainCookRecipeDataset(Dataset):
    def __init__(self, features_path):
        """
        Dataset per il task di Task Verification (Estensione).
        Carica le feature pre-calcolate a livello di ricetta.
        
        Args:
            features_path (str): Percorso al file .pkl creato nel Substep 1
                                 contenente dizionario {video_id: {'features': ..., 'label': ...}}
        """
        super().__init__()
        
        # Carichiamo il file generato nello script di preprocessing (Substep 1)
        # Struttura attesa del pickle: 
        # un elenco (list) di dizionari, oppure un dizionario di dizionari.
        # Qui assumo una lista di campioni per semplicità.
        with open(features_path, 'rb') as f:
            self.data = pickle.load(f)
            
        # Se self.data è un dizionario {video_id: content}, lo convertiamo in lista
        if isinstance(self.data, dict):
            self.data = list(self.data.values())

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
        
        # Recupera la label (0 = Corretto, 1 = Errato o viceversa in base alla tua convenzione)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        
        # Opzionale: ritorniamo anche l'ID per tracciare quale ricetta stiamo analizzando
        video_id = sample.get('video_id', str(idx))

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