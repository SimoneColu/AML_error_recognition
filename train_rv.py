import numpy as np
import torch
import wandb
import os
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.models.recipe_verifier import RecipeVerifier


def train_on_batch(model, data, optimizer, criterion, device):
    model.train()
    data = data.to(device)

    # TRICK: NOISE INJECTION
    # Adding gaussian noise to the features (Only during the training) to contrast OVERFITTING
    noise = torch.randn_like(data.x) * 0.02
    data.x = data.x + noise
    
    optimizer.zero_grad()

    logits = model(data)
    logits = logits.view(-1)
    target = data.y.float().view(-1)

    loss = criterion(logits, target)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    preds = (logits > 0).float()
    num_correct = (preds == target).sum().item()
    num_graphs = target.numel()

    return loss.item(), num_correct, num_graphs

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

    for data in train_loader:
        batch_loss, num_correct, num_graphs = train_on_batch(model, data, optimizer, criterion, device)

        total_loss += batch_loss * num_graphs
        total_correct += num_correct
        total_graphs += num_graphs

    #Loss and Accuracy are computed considerning graph instead of batches (weighted sum), so the avg_loss is more accurated. It avoid to consider small batches equal to others
    avg_loss = total_loss / total_graphs if total_graphs > 0 else 0.0
    accuracy = total_correct / total_graphs if total_graphs > 0 else 0.0

    return avg_loss, accuracy

def print_model_metrics(fold, oof_preds, oof_labels, oof_probs):
    """
    oof_preds: hard labels (0 or 1)
    oof_labels: ground truth labels
    oof_probs: raw probabilities/logits for Class 1
    """
    print(f"\n--- RESULTS AT FOLD {fold} ---")

    np_true = np.asarray(oof_labels).astype(int)
    np_pred = np.asarray(oof_preds).astype(int)
    np_probs = np.asarray(oof_probs)

    total_count = len(np_true)
    correct_count = (np_true == np_pred).sum()
    
    acc = correct_count / total_count if total_count > 0 else 0.0
    
    f1 = f1_score(np_true, np_pred, average='macro')
    
    try:
        auc = roc_auc_score(np_true, np_probs)
    except ValueError:
        auc = 0.5  # Default if only one class is present in the fold
        
    wandb.log({
        "Model_accuracy": acc,
        "Macro_F1": f1,
        "AUC": auc
    })

    print(f"PROCESSED GRAPHS: {total_count}")
    print(f"ACCURACY:  {acc:.4f}")
    print(f"MACRO F1:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"PRED DIST: {np.bincount(np_pred)}")
    print("----------------------------------------\n")

def get_probs(model, loader, device):
    model.eval()
    all_probs, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for batch in loader:
            data = batch.to(device)
            logits = model(data).view(-1)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.view(-1).cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_logits = np.concatenate(all_logits)

    preds = (all_probs >= 0.5).astype(int)
    return all_probs, all_labels, all_logits, preds

def find_optimal_threshold(all_probs, all_labels):
    best_threshold = 0.5
    best_f1 = 0
    
    # Test 100 different threshold values from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 100)
    
    for t in thresholds:
        # Apply current threshold
        preds = (all_probs >= t).astype(int)
        # Calculate Macro F1
        current_f1 = f1_score(all_labels, preds, average='macro')
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t
            
    # Final check on accuracy with the best threshold
    best_preds = (all_probs >= best_threshold).astype(int)
    best_acc = accuracy_score(all_labels, best_preds)
    
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Improved Macro F1: {best_f1:.4f}")
    print(f"Improved Accuracy: {best_acc:.4f}")
    
    return best_threshold


def compute_detailed_metrics(all_labels, all_preds):
    # 1. Macro-averaged metrics (treats both classes equally)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    
    # 2. Per-class metrics (0=Correct, 1=Error)
    # This is crucial for your "Discussion of Error Sensitivity"
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    
    print(f"PRECISION (Macro): {precision_macro:.4f}")
    print(f"RECALL (Macro):    {recall_macro:.4f}")
    print("-" * 30)
    print(f"CLASS 'Correct' (0) - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}")
    print(f"CLASS 'Error'   (1) - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}")
    
    return precision_macro, recall_macro

def run_cross_validation (dataset, strategy='logo', groups=None, **cfg):
    """
    Unified engine for LOGO and LOO Cross-Validation.
    strategy: 'logo' or 'loo'
    """
    
    # Create a directory for the per-fold metrics (derived from checkpoint_file name)
    # E.g. if checkpoint_file is "loo.pt", dir will be "loo_folds_metrics"
    preds_dir = cfg['checkpoint_file'].replace(".pt", "") + "_folds_metrics"
    os.makedirs(preds_dir, exist_ok=True)

    wandb.define_metric("epoch_in_fold")
    wandb.define_metric("train_loss_fold", step_metric="epoch_in_fold")
    wandb.define_metric("train_acc_fold", step_metric="epoch_in_fold")
    wandb.define_metric("folds_processed")
    wandb.define_metric("running_logo_accuracy", step_metric="folds_processed")

    splitter = LeaveOneGroupOut() if strategy == 'logo' else LeaveOneOut()

    all_oof_probs = []
    all_oof_labels = []
    all_oof_preds = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.arange(len(dataset)), groups=groups), 1):
        
        metric_path = os.path.join(preds_dir, f"fold_{fold}_metrics.pt")

        if os.path.exists(metric_path):
            saved_data = torch.load(metric_path)
            
            all_oof_labels.append(saved_data['label'])
            all_oof_preds.append(saved_data['pred'])
            all_oof_probs.append(saved_data['prob']) 
            
            oof_preds = np.concatenate(all_oof_preds)
            oof_labels = np.concatenate(all_oof_labels)
            oof_probs = np.concatenate(all_oof_probs)
            
            t_hist = saved_data.get('train_loss_history', [])
            if t_hist:
                for e_idx, t_l in enumerate(t_hist):
                    wandb.log({
                        "train_loss_fold": t_l,
                        "epoch_in_fold": e_idx,
                    })
            
            print(f"Fold {fold} found in cache. Replaying logs and skipping...")
            continue            
            
        print(f"\nFold: {fold}")
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=cfg['bs'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_subset, batch_size=cfg['bs'], shuffle=False, num_workers=2, pin_memory=True)

        model = RecipeVerifier(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout']
        ).to(cfg['device'])

        optimizer = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        criterion = nn.BCEWithLogitsLoss()
        
        fold_train_losses = []

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

        for ep in range(1, cfg['epochs'] + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg['device'])

            scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            fold_train_losses.append(train_loss)
            
            wandb.log({
                "epoch_in_fold": ep,
                "train_loss_fold": train_loss,
                "train_acc_fold": train_acc,
                "current_lr": current_lr
            })
            
            print(f"Epoch {ep:03d} | "f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | ")

        probs, labels, logits, preds = get_probs(model, val_loader, cfg['device'])

        all_oof_probs.append(probs)
        all_oof_preds.append(preds)
        all_oof_labels.append(labels)
        
        fold_metrics = {
            'fold': fold,
            'label': labels, 
            'pred': preds,
            'prob': probs,
            'logits': logits,
            'train_loss_history': fold_train_losses,
        }
        torch.save(fold_metrics, metric_path)
        print(f"--> [Checkpoint] Fold {fold} metrics saved.")

        if fold > 0 and fold % 5 == 0:
          tmp_preds  = np.concatenate(all_oof_preds)
          tmp_labels = np.concatenate(all_oof_labels)
          tmp_probs = np.concatenate(all_oof_probs)
          print_model_metrics(fold, tmp_preds, tmp_labels, tmp_probs)
          compute_detailed_metrics(tmp_preds, tmp_labels)

    oof_preds = np.concatenate(all_oof_preds)
    oof_labels = np.concatenate(all_oof_labels)
    oof_probs = np.concatenate(all_oof_probs)

    print_model_metrics(fold, oof_preds, oof_labels, oof_probs)
    compute_detailed_metrics(oof_preds, oof_labels)

    print("\n--- SEARCHING FOR OPTIMAL THRESHOLD ---")
    best_t = find_optimal_threshold(oof_probs, oof_labels)
    
    final_preds = (oof_probs >= best_t).astype(int)
    
    print("\n--- GLOBAL PERFORMANCE (OPTIMIZED THRESHOLD) ---")
    print_model_metrics(f"FINAL_OPT_{best_t:.2f}", final_preds, oof_labels, oof_probs)
    compute_detailed_metrics(oof_preds, oof_labels)

def train_final_model(dataset, **cfg):
    """
    Trains the model on the 100% of the dataset and saves the state_dict.
    Use this after CV to prepare the model for delivery.
    """
    run = wandb.init(
        project=cfg.get("project_name", "DAGNN_final_training"),
        name=cfg.get("run_name", "final_prod_run"),
        config=cfg
    )

    loader = DataLoader(dataset, batch_size=cfg['bs'], shuffle=True)

    model = RecipeVerifier(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg.get('hidden_dim', 64),
        num_layers=cfg.get('num_layers', 2),
        dropout=cfg.get('dropout', 0.5)
    ).to(cfg['device'])

    optimizer = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"Starting Final Training on {len(dataset)} graphs...")

    for ep in range(1, cfg['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, cfg['device'])
        
        if ep % 5 == 0:
            print(f"Epoch {ep:03d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            wandb.log({
                "final_epoch": ep,
                "final_loss": train_loss,
                "final_acc": train_acc
            })

    # Save the state_dict (Weights only)
    save_path = os.path.join(cfg.get('save_dir', './'), "recipe_verifier_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Successfully saved final model to: {save_path}")
    
    run.finish()
    return save_path