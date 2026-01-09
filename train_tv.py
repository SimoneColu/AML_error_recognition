## training for Task Verification (Extension)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import wandb
import numpy as np
from tqdm import tqdm

import os

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

    preds_dir = os.path.join(config.ckpt_directory, "loo_predictions")
    os.makedirs(preds_dir, exist_ok=True)

    # Liste globali
    global_y_true = []
    global_y_pred = []

    num_samples = len(full_dataset)
    print(f"Starting Leave-One-Out Cross Validation on {num_samples} recipes...")

    # 2. Loop Leave-One-Out: Itera su ogni ricetta
    # k è l'indice della ricetta che useremo come TEST in questa iterazione
    for k in tqdm(range(num_samples[:20]), desc="LOO Folds"):

        metric_path = os.path.join(preds_dir, f"fold_{k}_metrics.pt")
        
        # Se Abbiamo già i risultati finali (metriche) Skippiamo il fold
        if os.path.exists(metric_path):
            saved_data = torch.load(metric_path)
            global_y_true.append(saved_data['label'])
            global_y_pred.append(saved_data['pred'])
            continue

        # --- Data Splitting ---
        indices = list(range(num_samples))
        test_idx = [indices.pop(k)] # Rimuovi l'indice k e usalo per il test
        train_idx = indices         # Il resto è training
        
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)
        
        # DataLoader specifici per questo fold
        # Nota: batch_size basso per il training dato che sono pochi dati
        
        test_loader = DataLoader(test_subset, batch_size=1, 
                                 shuffle=False, collate_fn=recipe_collate_fn)
        
        # --- Model Initialization ---
        # Reinizializziamo il modello da zero ad ogni fold per non avere data leakage
        model = RecipeVerifier(config).to(config.device)
        
       
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, collate_fn=recipe_collate_fn)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        model.train()
        for epoch in range(config.num_epochs):
            for batch in train_loader:
                features, labels, masks, _ = batch
                features = features.to(config.device)
                labels = labels.to(config.device).unsqueeze(1)
                masks = masks.to(config.device)
                
                optimizer.zero_grad()
                outputs = model(features, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()


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

                val_label = labels.item()
                val_pred = preds.item()
                val_logit = logits.item()

                # Salviamo subito le metriche leggere per non dover rifare questo fold
                fold_metrics = {
                    'fold': k,
                    'label': val_label,
                    'pred': val_pred,
                    'logits': val_logit
                }
                torch.save(fold_metrics, metric_path)
                
                # 2. Accumula in memoria (per calcolo finale oggi)
                global_y_true.append(val_label)
                global_y_pred.append(val_pred)


    # --- Final Aggregation ---
    y_true_arr = np.array(global_y_true)
    y_pred_arr = np.array(global_y_pred)

    # Evita divisione per zero se liste vuote
    if len(y_true_arr) > 0:
        accuracy = np.mean(y_true_arr == y_pred_arr)
        correct_count = np.sum(y_true_arr == y_pred_arr)
    else:
        accuracy = 0.0
        correct_count = 0

    print(f"\nTask Verification Results:")
    print(f"Total Recipes Processed: {len(y_true_arr)}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Final Accuracy: {accuracy:.4f}")

    if config.enable_wandb:
        wandb.log({"tv_loo_accuracy": accuracy})

    
    # ==========================================
    # FASE 2: FINAL FULL TRAINING (Tutti i dati)
    # ==========================================
    print("\nStarting final training on ALL data (Production Model)...")
    
    # Dataset completo senza split
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=recipe_collate_fn)
    
    # Nuovo modello pulito
    final_model = RecipeVerifier(config).to(config.device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    final_model.train()
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        for batch in full_loader:
            features, labels, masks, _ = batch
            features = features.to(config.device)
            labels = labels.to(config.device).unsqueeze(1)
            masks = masks.to(config.device)
            
            optimizer.zero_grad()
            outputs = final_model(features, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if config.enable_wandb and epoch % 5 == 0:
             wandb.log({"final_training_loss": epoch_loss / len(full_loader)})

    # Salvataggio dell'UNICO modello finale
    final_ckpt_path = f"{config.ckpt_directory}/recipe_verifier_FULL_DATASET.pt"
    torch.save(final_model.state_dict(), final_ckpt_path)
    print(f"✅ Final model saved at: {final_ckpt_path}")
    
    if config.enable_wandb:
        wandb.save(final_ckpt_path)



def main():
    conf = Config()

    conf.task_name = const.TASK_VERIFICATION
    
    if conf.model_name is None:
        conf.model_name = "RecipeVerifier"

    if conf.enable_wandb:
        init_logger_and_wandb(conf)

    # Chiamata alla funzione specifica per l'estensione
    train_task_verification_loop(conf)

    if conf.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()