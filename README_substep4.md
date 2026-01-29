# Recipe Error Recognition with DAGNN

This repository implements a **Directed Acyclic Graph Neural Network (DAGNN)** designed to verify the correctness of recipe executions. By modeling task dependencies as a DAG, the system identifies logical errors, missing steps, or out-of-order operations in complex kitchen workflows.

## Project Objectives
The goal is to develop a robust classifier capable of distinguishing between correct and erroneous task sequences. Key focus areas include:
* **Topological Processing:** Respecting the inherent causal order of tasks.
* **Generalization:** Ensuring high performance on entirely new recipe types.
* **Error Sensitivity:** Evaluating the precision-recall trade-off for the "Error" class to minimize false negatives.

---

## Project Structure

The project is modularized to separate core logic from the execution environment:

### Core Logic (`/core`)
* **`recipe_verifier.py`**: Defines the `RecipeVerifier` architecture.
* **`topological_utils.py`**: Mathematical utilities to compute topological levels. This ensures node updates in the DAGNN follow the logical dependency flow.

* **`data_utils.py`**: Automates data ingestion, level injection for graphs, and group extraction for cross-validation.

### Training Engine (`train_rv.py`)
This script manages the model's training and evaluation pipeline:
* **Unified CV Engine:** Supports both **Leave-One-Out (LOO)** and **Leave-One-Group-Out (LOGO)** strategies via a single function call.
* **Metrics:** Computes comprehensive metrics to assess model accuracy and reliability.
* **Performance Tuning:** Includes logic to optimize decision thresholds and maximize final metrics.

### Entry Point
* **`execution_script.ipynb`**: The primary dashboard. It handles environment setup (cloning the repo, installing dependencies), hyperparameter configuration, and triggers the pipeline.

---

## Evaluation Strategies
To ensure statistical rigor, we utilize two cross-validation techniques:

| Strategy | Description | Purpose |
| :--- | :--- | :--- |
| **LOGO** | Leaves out all executions of a specific recipe (e.g., "Pasta"). | Evaluates generalization to **unseen recipe types**. |
| **LOO** | Leaves out a single execution instance. | Evaluates model robustness against **individual variations**. |

---

## Getting Started

### 1. Requirements
* PyTorch, PyTorch Geometric and Numpy
* Scikit-Learn
* Weights & Biases (for real-time experiment tracking)

### 2. Execution Flow
1. Open `execution_script.ipynb`.
2. Configure the `config` dictionary with your desired hyperparameters.
3. Run the cells to clone the repository and begin training.

```python
training_config = {
    "input_dim": dataset[0].x.size(1),
    "hidden_dim": 64,
    "num_layers": 2,
    "epochs": 30,
    "dropout": 0.5,
    "lr": 0.001,
    "wd": 0.005,
    "bs": 64,
    "device": device,
    "checkpoint_file": "logo_checkpoint.pt"
}