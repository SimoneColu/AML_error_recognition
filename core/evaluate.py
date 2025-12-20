import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from base import fetch_model, test_er_model, train_sub_step_test_step_dataset_base, train_model_base
from constants import Constants as const
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset, collate_fn


@dataclass
class Config(object):
    backbone: str = "omnivore"
    modality: str = "video"
    phase: str = "train"
    segment_length: int = 1
    # Use this for 1 sec video features
    segment_features_directory: str = "data/"

    ckpt_directory: str = "/data/rohith/captain_cook/checkpoints/"
    split: str = "recordings"
    batch_size: int = 1
    test_batch_size: int = 1
    ckpt: Optional[str] = None
    seed: int = 1000
    device: str = "cuda"

    variant: str = const.TRANSFORMER_VARIANT
    task_name: str = const.ERROR_RECOGNITION


def eval_er(config, threshold):
    model = fetch_model(config)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load the model from the ckpt file
    model.load_state_dict(torch.load(config.ckpt_directory))
    model.eval()

    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, collate_fn=collate_fn)

    # Calculate the evaluation metrics
    test_er_model(model, test_loader, criterion, config.device, phase="test", step_normalization=True, sub_step_normalization=True, threshold=threshold)

# Aggiunta funzione per trainare il modello su EGOVLP-----------------
def train_egovlp(config):
    # 1. Prepara i Dataloader
    # Usa la funzione che abbiamo visto nell'altro file. 
    # Questa prepara train (sub-step) e val/test (step-level)
    train_loader, val_loader, test_loader = train_sub_step_test_step_dataset_base(config)

    # --- BLOCCO DI VERIFICA ---
    print("Verifica Dimensioni Feature...")
    data_batch, target_batch, _ = next(iter(train_loader))
    print(f"Shape dell'input (Feature): {data_batch.shape}")
    print("uso la backbone: ", conf.backbone)

    # 2. Avvia il Training
    # Questa funzione gestisce epoche, loss, optimizer e salvataggio
    train_model_base(train_loader, val_loader, config, test_loader=test_loader)    
    # questa funzione fa tutto: train, validation e test
#-----------------------   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=[const.STEP_SPLIT, const.RECORDINGS_SPLIT], required=True)
    parser.add_argument("--backbone", type=str, choices=[const.SLOWFAST, const.OMNIVORE], required=True)
    parser.add_argument("--variant", type=str, choices=[const.MLP_VARIANT, const.TRANSFORMER_VARIANT], required=True)
    parser.add_argument("--phase", type=str, choices=[const.TEST, const.TRAIN], default=const.TEST) # Aggiunta fase train
    parser.add_argument("--modality", type=str, choices=[const.VIDEO])
    parser.add_argument("--ckpt", type=str, required=False) # Aggiunta: reso opzionale per training
    parser.add_argument("--threshold", type=float, required=False, default=0.5) # Aggiunta: reso opzionale per training
    args = parser.parse_args()

    # aggiunta parametri di training ---------------------------
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=50, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    # -----------------------------
    conf = Config()
    conf.split = args.split
    conf.backbone = args.backbone
    conf.variant = args.variant
    conf.phase = args.phase
    conf.modality = args.modality
    conf.ckpt_directory = args.ckpt

    # aggiunta nuova configurazione per training------------------ 
    # Imposta parametri di training nella config
    conf.lr = args.lr
    conf.num_epochs = args.epochs
    conf.batch_size = args.batch_size
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.task_name = const.ERROR_RECOGNITION 
    conf.seed = 42
    conf.enable_wandb = False # Metti True se vuoi il logging su WandB
    # Importante: alcune funzioni del file motore si aspettano weight_decay nella config
    conf.weight_decay = 1e-4
    # --------------------------------

    # aggiunta LOGICA DI SELEZIONE ---
    if conf.phase == const.TRAIN:
        print(f"Avvio TRAINING: Backbone={conf.backbone}, Variant={conf.variant}, Epochs={conf.num_epochs}")
        # Dove salvero i checkpoint
        conf.ckpt_directory = "checkpoint_egoVLP/" 
        train_egovlp(conf)
        
    elif conf.phase == const.TEST:
        if not args.ckpt:
            raise ValueError("Errore: Per la fase di TEST devi specificare --ckpt")
        conf.ckpt_directory = args.ckpt
        print(f"Avvio EVALUATION sul checkpoint: {conf.ckpt_directory}")
        eval_er(conf, args.threshold)
    # --------------------------------