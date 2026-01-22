## training for Task Verification (Extension)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import wandb
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau  # <--- IMPORTANTE

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
    Gestisce il training con strategia Leave-One-Out (LOO).
    """
    # ===================================
    # ========== Initial Setup ==========
    # ===================================

    
    # Set the path for the features from the arguments
    features_path = getattr(config, 'recipe_features_path', 'recipe_features.pkl') 
    
    # Set dir for saving checkpoints 
    preds_dir = os.path.join(config.ckpt_directory, "loo_predictions")
    os.makedirs(preds_dir, exist_ok=True)

    # Loading the full dataset
    full_dataset = CaptainCookRecipeDataset(features_path=features_path)

    # Global Lists
    global_y_true = []
    global_y_pred = []

    num_samples = len(full_dataset)

    # ===========================================
    # ========== Evaluation Loop (LOO) ==========
    # ===========================================

    print(f"Starting Leave-One-Out Cross Validation on {num_samples} recipes...")

    # We define a 'group' in wandb to easily filter this specific experiment run
    if config.enable_wandb:
        # --- Define Metrics for Overlapping Charts ---
        # This forces 'train_loss_fold' to always use 'epoch_in_fold' as its X-axis
        wandb.define_metric("epoch_in_fold")
        wandb.define_metric("train_loss_fold", step_metric="epoch_in_fold")
        
        # This forces the global accuracy to use the fold index as its X-axis
        wandb.define_metric("folds_processed")
        wandb.define_metric("running_loo_accuracy", step_metric="folds_processed")
    
    # k is the index of the recipe execution we are testing (so it will not be included in the training)
    for k in tqdm(range(num_samples), desc="LOO Folds"):
        
        # set the path to save the metrics for each fold
        metric_path = os.path.join(preds_dir, f"fold_{k}_metrics.pt")
        
        # Skipping logic if we already have the metrics for that fold
        if os.path.exists(metric_path):
            saved_data = torch.load(metric_path)
            global_y_true.append(saved_data['label'])
            global_y_pred.append(saved_data['pred'])

            if config.enable_wandb:
                curr_acc = np.mean(np.array(global_y_true) == np.array(global_y_pred))
                wandb.log({
                    "folds_processed": k,
                    "running_loo_accuracy": curr_acc
                })

            continue

        # --- Data Splitting ---
        indices = list(range(num_samples))
        test_idx = [indices.pop(k)] # Removing the k-index
        train_idx = indices         # the rest is the training
        
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)
        
        # --- DataLoader for this specific fold ---
        test_loader = DataLoader(
            test_subset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            collate_fn=recipe_collate_fn
            )
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            collate_fn=recipe_collate_fn
            )
        
        # --- Model Initialization ---
        # We need to re-initialize the model at each fold. Each fold trains a new model from scratch
        model = RecipeVerifier(config).to(config.device)
        
       # --- Optimizer and criterion definition ---
       # If Class 0 is 44% and Class 1 is 56%, weight should be ~1.27 for Class 0
        pos_weight = torch.tensor([1.27]).to(config.device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr = config.lr,
            weight_decay = config.weight_decay 
            )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # --- SCHEDULER IMPLEMENTATION ---
        # mode='min' because we are monitoring the loss (which needs to decrease)
        # Patience=5: number of epoch to wait before reducing LR
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',      
            factor=0.5,      # reducing factor
            patience=5, 
            min_lr=1e-7
        )

        train_loss_history = [] # to save it in ckpt, useful for debug
        
        # --- Model Training ---
        model.train()
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                features, labels, masks, _ = batch
                features = features.to(config.device)
                labels = labels.to(config.device).unsqueeze(1)
                masks = masks.to(config.device)
                
                optimizer.zero_grad()
                outputs = model(features, masks)
                loss = criterion(outputs, labels)

                # Check NaN 
                if torch.isnan(loss).any():
                    print(f"Loss NaN at fold {k}, epoch {epoch}. Skipping step.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            train_loss_history.append(avg_loss)

            # --- SCHEDULER STEP ---
            # We pass the loss for this epoch
            # He will decide to wether deacrease the loss for the next epoch
            scheduler.step(avg_loss)
            
            
            if config.enable_wandb:
                wandb.log({
                    "train_loss_fold": avg_loss,    # The Y-Axis
                    "epoch_in_fold": epoch,         # The X-Axis (0 to 10)
                    "fold_group_id": f"fold_{k}",   # The Grouping Key (Fold 1, Fold 2...)
                    "lr": optimizer.param_groups[0]['lr']
                })

         


        # --- Single Step Evaluation ---
        # Test on the k-th recipe
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                features, labels, masks, _ = batch
                features = features.to(config.device)
                labels = labels.to(config.device).unsqueeze(1)
                masks = masks.to(config.device)
                
                logits = model(features, masks)

                sigmoid_output = logits.sigmoid()
                
                # Binary prediction (Logits > 0 equals to Sigmoid > 0.5)
                preds = (sigmoid_output > 0.5).float()

                val_label = labels.item()
                val_pred = preds.item()
                val_logit = logits.item()

                # we save the metrics for this fold
                fold_metrics = {
                    'fold': k,
                    'label': val_label,
                    'pred': val_pred,
                    'logits': val_logit,
                    'prob': sigmoid_output,
                    'train_loss_history': train_loss_history,
                }
                torch.save(fold_metrics, metric_path)
                
                # save in memory the results for the final calculus
                global_y_true.append(val_label)
                global_y_pred.append(val_pred)
        
        running_acc = np.mean(np.array(global_y_true) == np.array(global_y_pred))

        if config.enable_wandb:
            wandb.log({
                "folds_processed": k,
                "running_loo_accuracy": running_acc
            })

        # --- Intermediate report ---
        if k > 0 and k % 1 == 0:
            print(f"\n--- Intermediate results at fold {k} ---")
            
            # Conversione sicura in array numpy
            np_true = np.array(global_y_true)
            np_pred = np.array(global_y_pred)
            
            correct_count = sum(1 for true, pred in zip(global_y_true, global_y_pred) if true == pred)
            total_count = len(global_y_true)
            
            acc = correct_count / total_count if total_count > 0 else 0.0
            
            # Pred distribution to show if our model is actually doing prediction or saying always the same label
            unique, counts = np.unique(np_pred, return_counts=True)
            pred_dist = {k.item(): v.item() for k, v in zip(unique, counts)}    

            print(f"Processed: {len(np_true)}")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Correct Prediction: {correct_count}")
            print(f"Pred Dist: {pred_dist}")
            print("----------------------------------------\n")

    # --- Final Aggregation ---
    y_true_arr = np.array(global_y_true)
    y_pred_arr = np.array(global_y_pred)

    # avoid zero-division
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

    
    # =========================================================
    # ========== Final Full Training with all Recipes =========
    # =========================================================
    
    print("\nStarting final training on ALL data (Production Model)...")
    
    # Full dataset, no splitt
    full_loader = DataLoader(full_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=recipe_collate_fn)
    
    # New clean model
    final_model = RecipeVerifier(config).to(config.device)
    
    # Optimizer and criterion definition
    optimizer = torch.optim.Adam(final_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
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

    # Save of the final model
    final_ckpt_path = f"{config.ckpt_directory}/recipe_verifier_FULL_DATASET.pt"
    torch.save(final_model.state_dict(), final_ckpt_path)
    print(f"Final model saved at: {final_ckpt_path}")
    
    if config.enable_wandb:
        wandb.save(final_ckpt_path)



def main():
    conf = Config()

    conf.print_config()

    conf.task_name = const.TASK_VERIFICATION
    
    if conf.model_name is None:
        conf.model_name = "RecipeVerifier"

    if conf.enable_wandb:
        init_logger_and_wandb(conf)

    
    train_task_verification_loop(conf)

    if conf.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()