# Files

## Notebooks

| File                  | Description             |
| --------------------- | ----------------------- |
| `prepare-data.ipynb`  | Split the dataset in smaller chunks to fit in GGDrive |
| `training-model.ipynb`| Train a model |
| `inference.ipynb`     | Make predictions with a trained model |

## Python files

| File | Description |
| - | - |
| `dataset.py` | Subclassing torch geometric dataset for our problem |
| `dynedge.py` | My implementation of the DynEdge architecture |
| `dynedge_global_var.py` | Same + add graph level features |
| `gnn_sage.py` | My first simple Graph Conv Network with Sage Convolutions |
| `metrics.py` | Metrics for performance analysis during model training |
| `pred_to_angles.py` | Conversion 2D/3D predictions |
