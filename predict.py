
import os
import time
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
PARAMETER_PATH = "models/best_hyperparameters.txt"

BINARY_THRESHOLD = 0.0


# =====================================================
# LOAD HYPERPARAMETERS
# =====================================================
def load_hyperparameters(filepath):

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Hyperparameter file not found: {filepath}"
        )

    params = {}

    with open(filepath, "r") as file:
        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if ":" in line:
                key, value = line.split(":", 1)

            elif "=" in line:
                key, value = line.split("=", 1)

            else:
                continue

            key = key.strip()
            value = value.strip().strip("\"'")

            if value.lower() in ("true", "false"):
                value = value.lower() == "true"

            elif value.lower() in ("none", "null"):
                value = None

            else:
                try:
                    if any(char in value.lower() for char in (".", "e")):
                        value = float(value)
                    else:
                        value = int(value)

                except ValueError:
                    pass

            params[key] = value

    required_params = [
        "hidden_dim",
        "num_gcn_layers",
        "dropout",
        "residual",
        "batch_norm",
        "activation",
        "use_bias",
    ]

    missing = [
        key for key in required_params
        if key not in params
    ]

    if missing:
        raise ValueError(
            f"Missing required hyperparameters in {filepath}: {missing}"
        )

    return params


# =====================================================
# BUILD DATASET FROM SINGLE PDB
# =====================================================
def build_dataset_from_single_pdb(pdb_file: str):

    pdb_file = os.path.abspath(pdb_file)
    pdb_dir = os.path.dirname(pdb_file)
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]

    csv_file = os.path.join(
        pdb_dir,
        f"{pdb_name}.csv"
    )

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Required CSV file not found: {csv_file}"
        )

    result = load_and_present_pdb(pdb_file)

    if result is None:
        return []

    node_features_tensor, protein_graphs, node_features_dict = result

    all_x = []
    src_list, dst_list, edge_weights = [], [], []

    node_offset = 0

    # =====================================================
    # NODE FEATURES AND EDGES
    # =====================================================
    for chain_id, G in protein_graphs.items():

        n = G.number_of_nodes()

        if n == 0:
            continue

        # NODE FEATURES
        x_chain = np.array([
            node_features_dict[chain_id][i]
            for i in range(n)
        ])

        scaler_x = StandardScaler()
        x_chain = scaler_x.fit_transform(x_chain)

        all_x.append(x_chain)

        # EDGES
        for u, v, attrs in G.edges(data=True):

            dist = attrs.get("distance", 15.0)
            weight = 1.0 / (dist + 1e-6)

            src_list.extend([
                u + node_offset,
                v + node_offset
            ])

            dst_list.extend([
                v + node_offset,
                u + node_offset
            ])

            edge_weights.extend([
                weight,
                weight
            ])

        node_offset += n

    if len(all_x) == 0:
        return []

    x = np.vstack(all_x)
    total_nodes = x.shape[0]

    x = torch.tensor(
        x,
        dtype=torch.float
    )

    # =====================================================
    # EDGE TENSOR
    # =====================================================
    if len(src_list) == 0:

        edge_index = torch.arange(
            total_nodes,
            dtype=torch.long
        ).repeat(2, 1)

        edge_attr = torch.ones(
            edge_index.size(1),
            dtype=torch.float
        )

    else:

        edge_index = torch.tensor(
            [src_list, dst_list],
            dtype=torch.long
        )

        edge_attr = torch.tensor(
            edge_weights,
            dtype=torch.float
        )

    # =====================================================
    # BUILD GRAPH
    # =====================================================
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    data.pdb_name = pdb_name

    data.residue_indices = torch.arange(
        total_nodes,
        dtype=torch.long
    )

    return [data]


# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_single_pdb(pdb_file, outdir=None, hyperparams=None):

    start_time = time.perf_counter()

    pdb_name = os.path.splitext(
        os.path.basename(pdb_file)
    )[0]

    print(f"Processing: {pdb_name}.pdb", flush=True)

    try:

        # -----------------------------
        # LOAD HYPERPARAMETERS
        # -----------------------------
        if hyperparams is None:
            hyperparams = load_hyperparameters(
                PARAMETER_PATH
            )

        pdb_file = os.path.abspath(pdb_file)
        folder = os.path.dirname(pdb_file)

        # -----------------------------
        # LOAD DATA
        # -----------------------------
        dataset = build_dataset_from_single_pdb(
            pdb_file
        )

        if len(dataset) == 0:
            raise ValueError("No graphs found")

        in_node_feats = dataset[0].x.size(1)

        # -----------------------------
        # LOAD MODEL
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
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=device,
                weights_only=True
            )
        )

        model.eval()

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

                batch_idx = (
                    batch.batch
                    if hasattr(batch, "batch")
                    else torch.zeros(
                        batch.x.size(0),
                        dtype=torch.long,
                        device=device
                    )
                )

                preds, _ = model(
                    batch.x,
                    batch.edge_index,
                    getattr(batch, "edge_attr", None),
                    batch_idx
                )

                all_preds.extend(
                    preds.cpu().numpy().flatten()
                )

        y_pred = np.array(all_preds)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        residue_indices = np.arange(
            1,
            len(y_pred) + 1
        )

        binary = (
            y_pred >= BINARY_THRESHOLD
        ).astype(int)

        if outdir is None:
            outdir = folder
        else:
            os.makedirs(
                outdir,
                exist_ok=True
            )

        out_csv = os.path.join(
            outdir,
            f"{pdb_name}_predictions.csv"
        )

        df = pd.DataFrame({
            "PDB_Name": pdb_name,
            "Residue_Index": residue_indices,
            "Predicted_RMSF_Norm": y_pred,
            "Predicted_Binary": binary
        })

        df.to_csv(
            out_csv,
            index=False
        )

    finally:

        elapsed_time = time.perf_counter() - start_time

        print(
            f"Finished: {pdb_name} | "
            f"Time: {elapsed_time:.2f} seconds",
            flush=True
        )


# =====================================================
# MAIN: PROCESS ONE OR MULTIPLE PDB FILES
# =====================================================
if __name__ == "__main__":

    import argparse
    import glob
    import time

    parser = argparse.ArgumentParser(
        description="Predict RMSF for one or multiple PDB files"
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="One or more PDB files or wildcard patterns"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None
    )

    args = parser.parse_args()

    # -------------------------------------------------
    # EXPAND WILDCARDS AND COLLECT PDB FILES
    # -------------------------------------------------
    pdb_files = []

    for item in args.input:

        matches = glob.glob(item)

        if matches:
            pdb_files.extend(matches)
        elif os.path.isfile(item):
            pdb_files.append(item)
        else:
            print(f"Skipping: no matching file found for {item}")

    # Keep only minimized PDB files and remove duplicates
    pdb_files = list(dict.fromkeys(
        os.path.abspath(pdb)
        for pdb in pdb_files
        if os.path.isfile(pdb)
        and os.path.basename(pdb).endswith("_min.pdb")
    ))

    if not pdb_files:
        raise SystemExit("No *_min.pdb files found.")

    # -------------------------------------------------
    # PROCESS FILES ONE BY ONE
    # -------------------------------------------------
    total_start = time.perf_counter()
    successful = 0
    failed = 0

    print(f"Found {len(pdb_files)} minimized PDB files.")

    for index, pdb_file in enumerate(pdb_files, start=1):

        print(
            f"\n[{index}/{len(pdb_files)}] "
            f"Processing: {os.path.basename(pdb_file)}",
            flush=True
        )

        try:
            predict_single_pdb(
                pdb_file,
                args.output
            )
            successful += 1

        except Exception as error:
            failed += 1
            print(
                f"Failed: {os.path.basename(pdb_file)} | {error}",
                flush=True
            )

    # -------------------------------------------------
    # TOTAL EXECUTION TIME
    # -------------------------------------------------
    total_elapsed = time.perf_counter() - total_start

    print("\n========== SUMMARY ==========")
    print(f"Total PDB files: {len(pdb_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print("=============================")
