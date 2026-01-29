# AML/DAAI 2025 - Task Verification With Transformer


## Replication Package

All the information required to reproduce the training process is available in the  
[`task-verification-transformer-replication.ipynb`](task-verification-transformer-replication.ipynb) notebook.

The notebook includes:
- Instructions for downloading the required dependencies  
- Environment setup details (the notebook was developed and tested in a **Kaggle environment**)  
- Cells to execute the full training pipeline  

### Branch Contribution
The contribution added by this branch are:
- `core/models/RecipeVerifier.py`:
    -  Implements the RecipeVerifier architecture using a Transformer Encoder backend.

- `dataloader/CaptainCookRecipeDataset.py`:
    - A custom PyTorch Dataset that maps pre-computed recipe features to labels from error_annotations.json.
 
- `train_tv`:
    - Implements a Leave-One-Out (LOO) Cross-Validation loop to provide reliable performance estimates on the recipe dataset.
    - Integrates Weights & Biases (WandB) for real-time tracking of running accuracy and loss curves across all folds.


### Training command

```bash
python train_tv.py \
  --model_name "task_verification_transformer" \
  --recipe_features_path "/kaggle/working/recipes_features.npy" \
  --ckpt_directory "/kaggle/working/checkpoints" \
  --num_epochs 30 \
  --batch_size 8 \
  --lr 1e-4 \
  --weight_decay 5e-2 \
  --dropout 0.3
```
**EXTRA**: An additional code snippet is provided in the notebook to analyze the results stored in `checkpoints`, allowing for the calculation of final aggregate accuracy and the visualization of average loss curves across all folds.
