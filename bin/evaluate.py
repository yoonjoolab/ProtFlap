import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================
# REGRESSION METRICS
# =========================
def compute_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# =========================
# EVALUATE PER RESIDUE
# =========================
def evaluate_and_plot_regression_per_residue(model, data_loader, device, load_rmsf_fn, fold_id=None, scaler_dict=None):
    """
    Evaluate regression per residue, produce metrics, CSV, and plots.
    Ensures all residues are included and raw RMSF predictions are used.
    """

    model.eval()
    all_preds, all_targets = [], []
    all_pdb_names, all_rmsf_values, all_residue_indices = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            for graph in batch.to_data_list():
                # Raw predictions from model (no activation squashing)
                preds, _ = model(
                    graph.x.to(device),
                    graph.edge_index.to(device),
                    getattr(graph, "edge_attr", None),
                    getattr(graph, "batch", None)
                )
                preds = preds.cpu().numpy().flatten()
                targets = graph.y.cpu().numpy().flatten()
                n_residues = len(targets)

                # Residue indices for this graph
                residue_indices = np.arange(n_residues)

                # PDB name per residue
                pdb_name = getattr(graph, "pdb_name", "Unknown")
                pdb_names_residues = [pdb_name] * n_residues if isinstance(pdb_name, str) else list(pdb_name)

                # Load original RMSF values from CSV
                rmsf_values = load_rmsf_fn(pdb_name)
                if rmsf_values is not None and len(rmsf_values) >= n_residues:
                    use_rmsf = rmsf_values[:n_residues]
                elif rmsf_values is not None:
                    use_rmsf = np.pad(rmsf_values, (0, n_residues - len(rmsf_values)), constant_values=np.nan)
                else:
                    use_rmsf = [None] * n_residues

                # Apply scaler inverse if provided
                if scaler_dict and pdb_name in scaler_dict:
                    scaler = scaler_dict[pdb_name]
                    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                    targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

                # Collect all data
                all_preds.extend(preds)
                all_targets.extend(targets)
                all_rmsf_values.extend(use_rmsf)
                all_pdb_names.extend(pdb_names_residues)
                all_residue_indices.extend(residue_indices)

    # =========================
    # METRICS
    # =========================
    rmse, mae, r2 = compute_regression_metrics(all_targets, all_preds)
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    # =========================
    # SAVE CSV (per residue)
    # =========================
    csv_name = f"regression_results_per_residue_fold{fold_id}.csv" if fold_id else "regression_results_per_residue.csv"
    with open(csv_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["PDB_Name", "Residue_Index", "True_RMSF", "Predicted_RMSF", "CSV_RMSF"])
        for pdb, res_idx, true_val, pred_val, rmsf_val in zip(all_pdb_names, all_residue_indices, all_targets, all_preds, all_rmsf_values):
            writer.writerow([pdb, res_idx, true_val, pred_val, rmsf_val])

    # =========================
    # SCATTER PLOT
    # =========================
    plt.figure(figsize=(6,6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.xlabel("True RMSF")
    plt.ylabel("Predicted RMSF")
    plt.title(f"True vs Predicted RMSF {'Fold ' + str(fold_id) if fold_id else ''}")
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')  # y=x reference
    plt.savefig(f"regression_scatter_fold{fold_id}.png" if fold_id else "regression_scatter.png")
    plt.close()

    # =========================
    # ERROR DISTRIBUTION
    # =========================
    errors = np.array(all_targets) - np.array(all_preds)
    plt.figure(figsize=(6,4))
    sns.histplot(errors, bins=50, kde=True)
    plt.xlabel("Prediction Error")
    plt.title(f"Error Distribution {'Fold ' + str(fold_id) if fold_id else ''}")
    plt.savefig(f"error_distribution_fold{fold_id}.png" if fold_id else "error_distribution.png")
    plt.close()

    return rmse, mae, r2
