---
name: lab-assignment-pipeline
description: Systematic approach to completing ML/lab assignments from a specification document into a runnable project with training, evaluation, and documentation
source: auto-skill
extracted_at: '2026-11-16T01:45:00.000Z'
---

## Procedure

When given a lab assignment document (PDF or similar) describing an ML project:

### 1. Understand requirements
- Read the assignment document to extract: goal, dataset requirements, architecture constraints, metrics, deliverables (code + report sections)
- Check the environment: Python version, available hardware (CPU/GPU/MPS), network access
- Create a `.venv` for reproducible execution

### 2. Choose dataset and architecture
- Pick a standard dataset (CIFAR-10, MNIST, etc.) that fits the assignment
- Select a simple but representative architecture (e.g., CNN for image classification)
- Confirm choices with the user before implementing

### 3. Create project structure
```
project/
├── .venv/                 # virtual environment
├── data/                  # downloaded dataset
├── models/                # saved model weights
├── plots/                 # evaluation charts
├── train.py               # training script (architecture + training loop)
├── evaluate.py            # evaluation script (metrics + charts)
├── requirements.txt       # dependencies
├── .gitignore             # exclude .venv/, data/, __pycache__/
└── README.md              # report matching assignment sections
```

### 4. Implement training script (`train.py`)
- Define the model class (architecture)
- Set up data loaders with train/val split
- Training loop with epoch logging (loss, accuracy)
- Save best model weights to `models/`
- Log training history to CSV

### 5. Implement evaluation script (`evaluate.py`)
- Load saved model
- Compute metrics on val and test sets
- Generate classification report (precision, recall, F1)
- Plot confusion matrix and training curves
- Save metrics to JSON

### 6. Run training and evaluation
- Use background tasks for long-running operations (model training, dataset download)
- Monitor progress via log files or file modification times
- Wait for completion before proceeding

### 7. Write the report (`README.md`)
Structure matches assignment requirements:
1. **Theoretical background** — explain key concepts (layers, loss, optimizer, metrics)
2. **System description** — architecture table, hyperparameters, data split, system requirements, project structure, run instructions
3. **Results** — training table, metrics table, plots, overfitting analysis, patterns observed
4. **Conclusions** — what was achieved, what could be improved
5. **References** — sources cited

### Key patterns
- **Background tasks** for long-running operations (training, downloading): use `is_background: true` to avoid blocking
- **Log buffering**: Python stdout may be buffered when redirected to file — monitor via `wc -c` on log file or file mtime
- **Epoch timing**: track model file mtime to estimate epoch duration when logs are buffered
- **Separation of concerns**: `train.py` handles training only, `evaluate.py` handles metrics only — both import shared model definition from `train.py`
- **Standard datasets**: use `torchvision.datasets` with `download=True` for automatic download
