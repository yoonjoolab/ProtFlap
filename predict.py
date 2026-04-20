# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

from sklearn.preprocessing import StandardScaler

from bin.train_val import load_and_present_pdb
from bin.gnn_model import NodeMLP_GCN


# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "models/best_model_cv.pth"
BINARY_THRESHOLD = 0.0

best_params = {
    "learning_rate": 0.0001,
    "dropout": 0.1667,
    "weight_decay": 0.0001,
    "batch_norm": False,
    "residual": True,
    "activation": "ELU",
    "use_bias": True,
    "hidden_dim": 128,
    "num_gcn_layers": 5
}


def build_dataset_from_single_pdb(pdb_file: str):

    pdb_file = os.path.abspath(pdb_file)
    pdb_dir = os.path.dirname(pdb_file)
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]

    csv_file = os.path.join(pdb_dir, f"{pdb_name}.csv")

    if not os.path.exists(csv_file):
        print(f"Skipping {pdb_name}: CSV file not found")
        return []

    result = load_and_present_pdb(pdb_file)
    if result is None:
        return []

    node_features_tensor, protein_graphs, node_features_dict = result

    all_x = []
    src_list, dst_list, edge_weights = [], [], []

    node_offset = 0

    # =====================================================
    # SAME STRUCTURE AS TRAINING LOOP
    # =====================================================
    for chain_id, G in protein_graphs.items():

        n = G.number_of_nodes()
        if n == 0:
            continue

        # -----------------------------
        # NODE FEATURES (IDENTICAL TO TRAINING)
        # -----------------------------
        x_chain = np.array([node_features_dict[chain_id][i] for i in range(n)])

        scaler_x = StandardScaler()
        x_chain = scaler_x.fit_transform(x_chain)

        all_x.append(x_chain)

        # -----------------------------
        # EDGES (IDENTICAL TO TRAINING)
        # -----------------------------
        for u, v, attrs in G.edges(data=True):

            dist = attrs.get("distance", 10.0)
            weight = 1.0 / (dist + 1e-6)

            src_list.extend([u + node_offset, v + node_offset])
            dst_list.extend([v + node_offset, u + node_offset])
            edge_weights.extend([weight, weight])

        node_offset += n

    if len(all_x) == 0:
        print(f"Skipping {pdb_name}: no valid node features")
        return []

    x = np.vstack(all_x)
    total_nodes = x.shape[0]

    x = torch.tensor(x, dtype=torch.float)

    # =====================================================
    # EDGE TENSOR (SAME LOGIC AS TRAINING)
    # =====================================================
    if len(src_list) == 0:
        edge_index = torch.arange(total_nodes, dtype=torch.long).repeat(2, 1)
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.float)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    # =====================================================
    # BUILD GRAPH (NO y)
    # =====================================================
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    data.pdb_name = pdb_name
    data.residue_indices = torch.arange(total_nodes, dtype=torch.long)

    return [data]

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_single_pdb(pdb_file, outdir=None, hyperparams=None):

    if hyperparams is None:
        hyperparams = best_params

    pdb_file = os.path.abspath(pdb_file)
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    folder = os.path.dirname(pdb_file)

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    dataset = build_dataset_from_single_pdb(pdb_file)

    print(f"📦 Loaded {len(dataset)} graphs for {pdb_name}")

    if len(dataset) == 0:
        raise ValueError("No graphs found")

    in_node_feats = dataset[0].x.size(1)

    # -----------------------------
    # MODEL
    # -----------------------------
    model = NodeMLP_GCN(
        in_node_feats,
        hidden_dim=hyperparams["hidden_dim"],
        num_gcn_layers=hyperparams["num_gcn_layers"],
        dropout=hyperparams["dropout"],
        use_residual=hyperparams["residual"],
        use_batch_norm=hyperparams["batch_norm"],
        activation=hyperparams["activation"],
        use_bias=hyperparams["use_bias"],
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )

    model.eval()
    print("✅ Model loaded")

    # -----------------------------
    # DATA LOADER
    # -----------------------------
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=Batch.from_data_list
    )

    # -----------------------------
    # INFERENCE
    # -----------------------------
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            batch_idx = batch.batch if hasattr(batch, "batch") else torch.zeros(
                batch.x.size(0), dtype=torch.long, device=device
            )

            preds, _ = model(
                batch.x,
                batch.edge_index,
                getattr(batch, "edge_attr", None),
                batch_idx
            )

            all_preds.extend(preds.cpu().numpy().flatten())

    y_pred = np.array(all_preds)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    residue_indices = np.arange(1, len(y_pred) + 1)
    binary = (y_pred >= BINARY_THRESHOLD).astype(int)

    if outdir is None:
        outdir = folder
    else:
        os.makedirs(outdir, exist_ok=True)

    out_csv = os.path.join(outdir, f"{pdb_name}_predictions.csv")

    df = pd.DataFrame({
        "PDB_Name": pdb_name,
        "Residue_Index": residue_indices,
        "Predicted_RMSF_Norm": y_pred,
        "Predicted_Binary": binary
    })

    df.to_csv(out_csv, index=False)

    print(f"✅ Saved predictions → {out_csv}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Predict RMSF from PDB")

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    predict_single_pdb(args.input, args.output)
