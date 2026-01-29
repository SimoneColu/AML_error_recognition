# AML/DAAI 2025 - Task Verification With Transformer


## Replication Package

All the information required to reproduce the training process is available in the  
[`task-verification-transformer-replication.ipynb`](task-verification-transformer-replication.ipynb) notebook.

The notebook includes:
- Instructions for downloading the required dependencies  
- Environment setup details (the notebook was developed and tested in a **Kaggle environment**)  
- Cells to execute the full training pipeline  

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
**EXTRA**: An additional code snippet is provided in the notebook to analyze the previously obtained results.
