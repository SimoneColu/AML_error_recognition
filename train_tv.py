## training for Task Verification (Extension)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import wandb
import numpy as np
from tqdm import tqdm

# Importa i moduli base esistenti (per coerenza e utilità comuni)
from base import fetch_model_name
from core.config import Config
from core.utils import init_logger_and_wandb
from constants import Constants as const

# Importa i NUOVI moduli specifici per l'estensione
from core.models.recipe_verifier import RecipeVerifier
from dataloader.CaptainCookRecipeDataset import CaptainCookRecipeDataset, recipe_collate_fn

def train_task_verification_loop(config):
    """
    Gestisce il training con strategia Leave-One-Out (LOO) come richiesto dal documento.
    """
    # 1. Carica il dataset completo (tutte le ricette)
    # Assumiamo che config abbia un campo per il path, altrimenti mettilo hardcoded o nei constants
    features_path = getattr(config, 'recipe_features_path', 'recipe_features.pkl') 
    # 1. Dataset completo
    full_dataset = CaptainCookRecipeDataset(features_path=features_path)

    # ---------------- DEBUG MODE ----------------
    DEBUG = False            # False per run completo
    DEBUG_NUM_RECIPES = 10  # 10–20 ideale

    if DEBUG:
        indices = list(range(min(DEBUG_NUM_RECIPES, len(full_dataset))))
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"[DEBUG MODE] Using only {len(full_dataset)} recipes")
    # --------------------------------------------

    num_samples = len(full_dataset)

    results = []
    print(f"Starting Leave-One-Out Cross Validation on {num_samples} recipes...")

    # 2. Loop Leave-One-Out: Itera su ogni ricetta
    # k è l'indice della ricetta che useremo come TEST in questa iterazione
    for k in tqdm(range(num_samples), desc="LOO Folds"):
        
        # --- Data Splitting ---
        indices = list(range(num_samples))
        test_idx = [indices.pop(k)] # Rimuovi l'indice k e usalo per il test
        train_idx = indices         # Il resto è training
        
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)
        
        # DataLoader specifici per questo fold
        # Nota: batch_size basso per il training dato che sono pochi dati
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, 
                                  shuffle=True, collate_fn=recipe_collate_fn)
        test_loader = DataLoader(test_subset, batch_size=1, 
                                 shuffle=False, collate_fn=recipe_collate_fn)
        
        # --- Model Initialization ---
        # Reinizializziamo il modello da zero ad ogni fold per non avere data leakage
        model = RecipeVerifier(config).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        # --- Inner Training Loop ---
        # Addestriamo per N epoche su (K-1) ricette
        model.train()
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                features, labels, masks, _ = batch
                
                features = features.to(config.device)
                labels = labels.to(config.device).unsqueeze(1) # [Batch, 1]
                masks = masks.to(config.device)
                
                optimizer.zero_grad()
                outputs = model(features, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                
                # --- LOG BATCH ---
                if config.enable_wandb:
                    wandb.log({
                        "fold": k,
                        "epoch": epoch,
                        "batch_loss": loss.item()
                    })
            # --- LOG EPOCH ---
            if config.enable_wandb:
                wandb.log({
                    "fold": k,
                    "epoch": epoch,
                    "epoch_loss": epoch_loss / len(train_loader)
                })
        
        fold_ckpt = f"{config.ckpt_directory}/recipe_verifier_fold{k}.pt"
        torch.save(model.state_dict(), fold_ckpt)
        print(f"Saved model weights for fold {k} at {fold_ckpt}")
        
        # --- Single Step Evaluation ---
        # Testiamo sulla k-esima ricetta
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                features, labels, masks, _ = batch
                features = features.to(config.device)
                labels = labels.to(config.device).unsqueeze(1)
                masks = masks.to(config.device)
                
                logits = model(features, masks)
                
                # Predizione binaria (Logits > 0 equivale a Sigmoid > 0.5)
                preds = (logits > 0).float()
                
                is_correct = (preds == labels).item()
                results.append(is_correct)

    # --- Final Aggregation ---
    accuracy = sum(results) / len(results)
    print(f"\nTask Verification Results:")
    print(f"Total Recipes: {len(results)}")
    print(f"Correct Predictions: {sum(results)}")
    print(f"Final Accuracy: {accuracy:.4f}")

    if config.enable_wandb:
        wandb.log({"tv_loo_accuracy": accuracy})

    torch.save(model.state_dict(), f"{config.ckpt_directory}/recipe_verifier_final.pt")


def main():
    conf = Config()
    # Possiamo definire un nome task custom se necessario in constants, 
    # oppure usare una stringa libera
    conf.task_name = const.TASK_VERIFICATION
    
    if conf.model_name is None:
        # Nota: fetch_model_name potrebbe aspettarsi task standard, 
        # potresti dover settare un nome manuale se dà errore
        conf.model_name = "RecipeVerifier"

    if conf.enable_wandb:
        init_logger_and_wandb(conf)

    # Chiamata alla funzione specifica per l'estensione
    train_task_verification_loop(conf)

    if conf.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()