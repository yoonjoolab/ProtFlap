import numpy as np
import pandas as pd
import torch
from .data_utils import load_rmsf_data


def get_predictions(model, loader, device="cpu"):
    """
    Run model inference for regression and return:
    - True RMSF values
    - Predicted RMSF values
    - PDB name for each residue
    - Residue index for each residue
    """

    model.eval()
    y_true, y_pred = [], []
    pdb_names, residue_indices = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            preds, _ = model(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                getattr(batch, "batch", None)
            )

            batch_targets = batch.y.cpu().numpy().flatten()
            batch_preds = preds.cpu().numpy().flatten()

            y_true.extend(batch_targets)
            y_pred.extend(batch_preds)

            # Handle batched graphs correctly
            ptr = batch.ptr.cpu().numpy()

            # residue indices must already be stored in Data object
            if hasattr(batch, "residue_indices") and batch.residue_indices is not None:
                batch_residue_indices = batch.residue_indices.cpu().numpy()
            else:
                batch_residue_indices = np.arange(len(batch_targets))

            batch_pdb_names = getattr(batch, "pdb_name", "Unknown")

            # PyG usually keeps pdb_name as a list for batched graphs
            if not isinstance(batch_pdb_names, (list, tuple, np.ndarray)):
                batch_pdb_names = [batch_pdb_names]

            for i in range(len(batch_pdb_names)):
                start = ptr[i]
                end = ptr[i + 1]
                n_nodes = end - start

                pdb_names.extend([batch_pdb_names[i]] * n_nodes)
                residue_indices.extend(batch_residue_indices[start:end].tolist())

    return (
        np.array(y_true),
        np.array(y_pred),
        pdb_names,
        residue_indices
    )


def compare_rmsf_and_predictions(pdb_names, residue_indices, y_true, y_pred):
    """
    Create a CSV-ready DataFrame with:
    - Actual RMSF from CSV
    - Model target RMSF
    - Predicted RMSF
    - One row per residue per PDB
    """
    results = []

    for pdb_name, res_idx, true_val, pred_val in zip(pdb_names, residue_indices, y_true, y_pred):
        rmsf_values = load_rmsf_data(pdb_name)

        csv_val = None
        if rmsf_values is not None and isinstance(rmsf_values, (list, np.ndarray)):
            if 0 <= int(res_idx) < len(rmsf_values):
                csv_val = float(rmsf_values[int(res_idx)])
            else:
                print(
                    f"Warning: Residue index {res_idx} out of bounds for "
                    f"{pdb_name} (length {len(rmsf_values)})"
                )
        else:
            print(f"Warning: load_rmsf_data for {pdb_name} returned invalid data.")

        results.append({
            "PDB_Name": pdb_name,
            "Residue_Index": int(res_idx),
            "Actual_RMSF_CSV": csv_val,
            "Target_RMSF": float(true_val),
            "Predicted_RMSF": float(pred_val)
        })

    df = pd.DataFrame(results)
    df.sort_values(by=["PDB_Name", "Residue_Index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
