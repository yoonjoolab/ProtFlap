import os
import csv
import logging
import subprocess
import re
import shutil
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import networkx as nx
import torch

# PyTorch
from torch.utils.data import Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PyTorch Geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, GATConv

# Sklearn (Regression)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Bio / Features
from Bio import PDB
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from aaindex import aaindex1

# ✅ Your modules (cleaned)
from .gnn_model import NodeMLP_GCN
from .data_utils import load_rmsf_data
from .prediction_utils import get_predictions, compare_rmsf_and_predictions

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

def count_flexible_residues(dataset, threshold=0.0):
    flexible_count, non_flexible_count = 0, 0
    for data in dataset:
        rmsf_values = load_rmsf_data(data.pdb_name)
        if rmsf_values is None:
            continue
        for val in rmsf_values:
            if val > threshold:
                flexible_count += 1
            else:
                non_flexible_count += 1
    return flexible_count, non_flexible_count


# ADD THESE FUNCTIONS HERE, BEFORE get_predictions
def parse_freesasa_output(output):
    lines = output.strip().split("\n")
    parsed_data = {}

    for line in lines:
        if line.startswith("#") or not line.strip():
            continue  # Skip header lines or empty lines

        # Example line: SEQ A    1  GLN :  196.95
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 5:
            try:
                residue_number = int(re.sub(r"\D", "", parts[2]))  # Remove non-digit characters
                surface_area = float(parts[5].replace(":", "").strip())
                parsed_data[residue_number] = surface_area
            except ValueError:
                continue  # Skip lines with invalid residue numbers

    return parsed_data

# Constants
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
    'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

AMINO_ACIDS = 'GAVLMIWFYSTCPNQKRHDE'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
POLARITY_CATEGORIES = {
    'nonpolar': ['A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W'],
    'polar_uncharged': ['C', 'N', 'Q', 'S', 'T', 'Y'],
    'polar_positive': ['H', 'K', 'R'],
    'polar_negative': ['D', 'E']
}

def one_hot_encode(residue_name):
    index = AA_TO_INDEX.get(residue_name, -1)
    if index != -1:
        encoding = [0] * 20
        encoding[index] = 1
        return encoding
    else:
        return [0] * 20

def get_polarity_encoding(residue_name):
    for category, residues in POLARITY_CATEGORIES.items():
        if residue_name in residues:
            return [1 if category == cat else 0 for cat in POLARITY_CATEGORIES.keys()]
    return [0] * len(POLARITY_CATEGORIES)

def dssp_simplified_encode(dssp_value):
    # S: Helix (H, G, I)
    # E: Extended (E, B)
    # L: Loop (T, S, -, C)
    if dssp_value in ['H', 'G', 'I']:
        return [1, 0, 0]  # Helix
    elif dssp_value in ['E', 'B']:
        return [0, 1, 0]  # Extended
    else:
        return [0, 0, 1]  # Loop (including coil, bend, turn, and undefined)

def get_residue_features(residue, freesasa_value, atomic_energies, dssp_value):
    residue_name = THREE_TO_ONE.get(residue.get_resname(), 'X')

    # One-hot encode residue features
    one_hot = one_hot_encode(residue_name)
    polarity_encoding = get_polarity_encoding(residue_name)

    # Other residue features from aaindex1
    hydrophobicity_values = aaindex1['KYTJ820101']['values']
    size_values = aaindex1['FASG760101']['values']
    charge_values = aaindex1['KLEP840101']['values']

    hydrophobicity = hydrophobicity_values.get(residue_name, 0)
    size = size_values.get(residue_name, 0)
    charge = charge_values.get(residue_name, 0)

    # Apply FreeSASA threshold: 1 if >= 15 else 0
    exposure = 1 if freesasa_value >= 15 else 0

    # DSSP structural encoding
    dssp_encoding = dssp_simplified_encode(dssp_value)

    # Combine all features into a single vector
    feature_vector = one_hot + polarity_encoding + [
        hydrophobicity, size, charge, exposure
    ] + dssp_encoding + atomic_energies

    return feature_vector

def move_mismatched_files(pdb_file, csv_file, destination_directory="mismatched_files"):
    """Moves the specified PDB and CSV files to the destination directory."""
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    pdb_name = os.path.basename(pdb_file)
    csv_name = os.path.basename(csv_file)
    
    pdb_destination = os.path.join(destination_directory, pdb_name)
    csv_destination = os.path.join(destination_directory, csv_name)

    try:
        shutil.move(pdb_file, pdb_destination)
        logging.info(f"Moved {pdb_name} to {destination_directory}")
    except Exception as e:
        logging.error(f"Error moving {pdb_name} to {destination_directory}: {e}")
        
    try:
        shutil.move(csv_file, csv_destination)
        logging.info(f"Moved {csv_name} to {destination_directory}")
    except Exception as e:
        logging.error(f"Error moving {csv_name} to {destination_directory}: {e}")

def load_and_present_pdb(pdb_file):
    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    atomic_energy_csv = f"{pdb_name}.csv"

    if not os.path.exists(atomic_energy_csv):
        logging.error(f"Required CSV file not found for {pdb_name}: {atomic_energy_csv}")
        return None

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("structure", pdb_file)
    except Exception as e:
        logging.error(f"PDB parsing error for {pdb_name}: {e}")
        move_mismatched_files(pdb_file, atomic_energy_csv)
        return None

    try:
        atomic_energy_data = pd.read_csv(atomic_energy_csv)
    except Exception as e:
        logging.error(f"Error reading atomic energy CSV for {pdb_name}: {e}")
        move_mismatched_files(pdb_file, atomic_energy_csv)
        return None

    # Run freesasa and parse output
    freesasa_command = f"freesasa --foreach-residue --no-log {pdb_file}"
    try:
        freesasa_result = subprocess.run(freesasa_command, shell=True, capture_output=True, text=True, check=True)
        freesasa_output = freesasa_result.stdout
        freesasa_dict = parse_freesasa_output(freesasa_output)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running freesasa for {pdb_name}: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"Error parsing freesasa output for {pdb_name}: {e}")
        return None

    # Clean and convert 'Residue' column in atomic_energy_data
    def clean_and_convert_residue(residue):
        residue = str(residue).strip().replace(',', '')
        try:
            return int(residue)
        except ValueError:
            return None

    atomic_energy_data["Residue"] = atomic_energy_data["Residue"].apply(clean_and_convert_residue)
    atomic_energy_data = atomic_energy_data.dropna(subset=["Residue"])
    atomic_energy_dict = dict(zip(atomic_energy_data["Residue"], atomic_energy_data.iloc[:, 1:29].values.tolist()))

    dssp_dict = dssp_dict_from_pdb_file(pdb_file)[0]

    all_node_features = []
    protein_graphs = {}
    node_features_dict = {}
    residue_counter = 0
    pdb_residue_count = 0

    for model in structure:
        for chain in model:
            residues_list = [residue for residue in chain if PDB.is_aa(residue)]
            chain_node_features = []
            chain_node_features_dict = {}
            protein_graph = create_protein_graph(structure, chain.get_id())
            protein_graphs[chain.get_id()] = protein_graph

            for idx, residue in enumerate(residues_list):
                pdb_residue_count += 1
                try:
                    residue_id = residue.get_id()
                    residue_number = residue_id[1]
                    freesasa_value = freesasa_dict.get(residue_number, 0)
                    atomic_energies = atomic_energy_dict.get(residue_number, [0] * 28)

                    dssp_value = dssp_dict.get((chain.get_id(), residue_id), '-')
                    features = get_residue_features(residue, freesasa_value, atomic_energies, dssp_value)
                    chain_node_features.append(features)
                    chain_node_features_dict[residue_counter] = features
                    residue_counter += 1
                except Exception as e:
                    logging.warning(f"Error processing residue {residue_id}: {e}")
                    continue
            all_node_features.extend(chain_node_features)
            node_features_dict[chain.get_id()] = chain_node_features_dict
            
    csv_residue_count = len(atomic_energy_data)

    if not all_node_features:
        logging.error(f"No valid residues found in {pdb_name}")
        move_mismatched_files(pdb_file, atomic_energy_csv)
        return None
        
    if pdb_residue_count != csv_residue_count:
        logging.warning(f"Mismatch in number of residues for {pdb_name}: PDB has {pdb_residue_count}, CSV has {csv_residue_count}")
        move_mismatched_files(pdb_file, atomic_energy_csv)
        return None

    node_features_tensor = torch.tensor(all_node_features, dtype=torch.float)
    logging.info(f"Processed {pdb_name}")
    print(f"Processed PDB: {pdb_name}, Overall node dimension: {node_features_tensor.shape[1]}")
    return node_features_tensor, protein_graphs, node_features_dict


def create_protein_graph(structure, chain_id, max_distance=10.0):
    graph = nx.Graph()
    chain = structure[0][chain_id]
    residues = [residue for residue in chain if PDB.is_aa(residue)]
    for idx, residue in enumerate(residues):
        graph.add_node(idx, residue=residue)
    for i, res1 in enumerate(residues):
        for j, res2 in enumerate(residues[i+1:], start=i+1):
            distance = np.linalg.norm(res1['CA'].coord - res2['CA'].coord)
            if distance < max_distance:
                graph.add_edge(i, j, distance=distance)
    return graph
    
def build_dataset_from_folder(folder: str):
    data_list = []

    for filename in os.listdir(folder):
        if not filename.endswith(".pdb"):
            continue

        pdb_path = os.path.join(folder, filename)
        pdb_name = os.path.splitext(filename)[0]

        result = load_and_present_pdb(pdb_path)
        if result is None:
            continue

        node_features_tensor, protein_graphs, node_features_dict = result

        csv_file = os.path.join(folder, f"{pdb_name}.csv")
        if not os.path.exists(csv_file):
            print(f"Skipping {pdb_name}: CSV file not found")
            continue

        df = pd.read_csv(csv_file)
        if "rmsf_norm" not in df.columns:
            print(f"Skipping {pdb_name}: 'rmsf_norm' column not found")
            continue

        # target already normalized
        rmsf = df["rmsf_norm"].values.astype(np.float32)

        all_x = []
        all_residue_indices = []
        src_list, dst_list, edge_weights = [], [], []

        node_offset = 0

        for chain_id, G in protein_graphs.items():
            n = G.number_of_nodes()
            if n == 0:
                continue

            # node features for this chain
            x_chain = np.array([node_features_dict[chain_id][i] for i in range(n)])

            # scale node features
            scaler_x = StandardScaler()
            x_chain = scaler_x.fit_transform(x_chain)

            all_x.append(x_chain)

            # residue indices for this chain in the final merged graph
            all_residue_indices.extend(range(node_offset, node_offset + n))

            # use edges already present in the graph
            for u, v, attrs in G.edges(data=True):
                dist = attrs.get("distance", 10.0)
                weight = 1.0 / (dist + 1e-6)

                src_list.extend([u + node_offset, v + node_offset])
                dst_list.extend([v + node_offset, u + node_offset])
                edge_weights.extend([weight, weight])

            node_offset += n

        if len(all_x) == 0:
            print(f"Skipping {pdb_name}: no valid node features")
            continue

        x = np.vstack(all_x)
        total_nodes = x.shape[0]

        # require exact residue match
        if total_nodes != len(rmsf):
            print(
                f"Skipping {pdb_name}: residue mismatch "
                f"(graph={total_nodes}, rmsf={len(rmsf)})"
            )
            continue

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(rmsf, dtype=torch.float)

        if len(src_list) == 0:
            edge_index = torch.arange(total_nodes, dtype=torch.long).repeat(2, 1)
            edge_attr = torch.ones(edge_index.size(1), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

        data.pdb_name = pdb_name
        data.residue_indices = torch.arange(total_nodes, dtype=torch.long)

        data_list.append(data)
        print(f"Added {pdb_name}: {total_nodes} residues")

    return data_list
    
def train_and_evaluate(trial, data_list):

    lr = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    dropout = trial.suggest_float("dropout", 0.001, 0.3)
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2])
    batch_norm = trial.suggest_categorical("batch_norm", [True, False])
    residual = trial.suggest_categorical("residual", [True, False])
    activation = trial.suggest_categorical("activation", ["ReLU", "ELU"])
    use_bias = trial.suggest_categorical("use_bias", [True, False])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_gcn_layers = trial.suggest_int("num_gcn_layers", 2, 6)

    n = len(data_list)
    n_train = int(0.9 * n)

    train_ds, val_ds = random_split(data_list, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=Batch.from_data_list)

    model = NodeMLP_GCN(
        data_list[0].x.size(1),
        hidden_dim=hidden_dim,
        num_gcn_layers=num_gcn_layers,
        dropout=dropout,
        use_residual=residual,
        use_batch_norm=batch_norm,
        activation=activation,
        use_bias=use_bias,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ✅ REGRESSION LOSS
    criterion = torch.nn.SmoothL1Loss()

    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(100):
        model.train()
        total_loss = 0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            preds, _ = model(batch.x, batch.edge_index)

            loss = criterion(preds, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes

        train_loss = total_loss / total_nodes

        # === VALIDATION ===
        model.eval()
        total_val_loss, total_val_nodes = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds, _ = model(batch.x, batch.edge_index)

                loss = criterion(preds, batch.y)
                total_val_loss += loss.item() * batch.num_nodes
                total_val_nodes += batch.num_nodes

        val_loss = total_val_loss / total_val_nodes

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val_loss

def train_one_fold(
    fold_id, train_idx, val_idx, dataset, best_params,
    device="cpu", epochs=100, batch_size=16, patience=20
):
    # ===== DATA LOADERS =====
    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Batch.from_data_list
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Batch.from_data_list
    )

    in_node_feats = dataset[0].x.size(1)

    # ===== MODEL =====
    model = NodeMLP_GCN(
        in_node_feats,
        hidden_dim=best_params["hidden_dim"],
        num_gcn_layers=best_params["num_gcn_layers"],
        dropout=best_params["dropout"],
        use_residual=best_params["residual"],
        use_batch_norm=best_params["batch_norm"],
        activation=best_params["activation"],
        use_bias=best_params["use_bias"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )

    criterion = torch.nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=10
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    # store best epoch outputs
    best_epoch_preds = None
    best_epoch_targets = None
    best_epoch_pdb_names = None
    best_epoch_residue_indices = None
    best_epoch_metrics = None

    # ===== TRAINING LOOP =====
    for epoch in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        total_train_loss = 0.0
        total_train_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            preds, _ = model(batch.x, batch.edge_index)
            loss = criterion(preds, batch.y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch.num_nodes
            total_train_nodes += batch.num_nodes

        train_loss = total_train_loss / total_train_nodes
        train_losses.append(train_loss)

        # ===== VALIDATION =====
        model.eval()
        total_val_loss = 0.0
        total_val_nodes = 0

        all_preds = []
        all_targets = []
        pdb_names = []
        residue_indices = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                preds, _ = model(batch.x, batch.edge_index)
                loss = criterion(preds, batch.y)

                total_val_loss += loss.item() * batch.num_nodes
                total_val_nodes += batch.num_nodes

                batch_preds = preds.detach().cpu().numpy().flatten()
                batch_targets = batch.y.detach().cpu().numpy().flatten()

                all_preds.extend(batch_preds)
                all_targets.extend(batch_targets)

                # ===== CORRECT GRAPH-WISE MAPPING =====
                ptr = batch.ptr.cpu().numpy()

                if hasattr(batch, "residue_indices") and batch.residue_indices is not None:
                    batch_residue_indices = batch.residue_indices.cpu().numpy()
                else:
                    batch_residue_indices = np.arange(len(batch_targets))

                batch_pdb_names = getattr(batch, "pdb_name", "Unknown")
                if not isinstance(batch_pdb_names, (list, tuple, np.ndarray)):
                    batch_pdb_names = [batch_pdb_names]

                for i in range(len(batch_pdb_names)):
                    start = ptr[i]
                    end = ptr[i + 1]
                    n_nodes_graph = end - start

                    pdb_names.extend([batch_pdb_names[i]] * n_nodes_graph)
                    residue_indices.extend(batch_residue_indices[start:end].tolist())

        val_loss = total_val_loss / total_val_nodes
        val_losses.append(val_loss)

        # ===== METRICS =====
        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)

        rmse = np.sqrt(mean_squared_error(all_targets_np, all_preds_np))
        mae = mean_absolute_error(all_targets_np, all_preds_np)
        r2 = r2_score(all_targets_np, all_preds_np)

        scheduler.step(val_loss)

        print(
            f"[Fold {fold_id}] Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}"
        )

        # ===== SAVE BEST =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0

            best_epoch_preds = all_preds_np.copy()
            best_epoch_targets = all_targets_np.copy()
            best_epoch_pdb_names = pdb_names.copy()
            best_epoch_residue_indices = residue_indices.copy()
            best_epoch_metrics = (rmse, mae, r2)
        else:
            patience_counter += 1

        # ===== EARLY STOPPING =====
        if patience_counter >= patience:
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch}")
            break

    # safety check
    if best_model_state is None:
        raise ValueError(f"No best model state was saved for fold {fold_id}")

    # ===== LOAD AND SAVE BEST MODEL =====
    model.load_state_dict(best_model_state)
    model_path = f"best_model_fold{fold_id}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[Fold {fold_id}] Best Val Loss: {best_val_loss:.4f}")

    # ===== SAVE CSV PER RESIDUE =====
    df = compare_rmsf_and_predictions(
        best_epoch_pdb_names,
        best_epoch_residue_indices,
        best_epoch_targets,
        best_epoch_preds
    )
    df.to_csv(f"fold_{fold_id}_predictions.csv", index=False)

    # ===== SAVE METRICS =====
    best_rmse, best_mae, best_r2 = best_epoch_metrics
    with open(f"fold_{fold_id}_metrics.txt", "w") as f:
        f.write(f"RMSE: {best_rmse:.4f}\n")
        f.write(f"MAE: {best_mae:.4f}\n")
        f.write(f"R2: {best_r2:.4f}\n")

    # ===== PLOTS =====
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title(f"Loss Curve Fold {fold_id}")
    plt.savefig(f"loss_curve_fold{fold_id}.png")
    plt.close()

    plt.figure()
    plt.scatter(best_epoch_targets, best_epoch_preds, alpha=0.5)
    plt.xlabel("Target RMSF")
    plt.ylabel("Predicted RMSF")
    plt.title(f"Regression Scatter Fold {fold_id}")
    plt.savefig(f"scatter_fold{fold_id}.png")
    plt.close()

    return best_val_loss, model_path, best_epoch_metrics
    
def run_cross_validation(dataset, best_params, n_splits=10):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):

        print(f"\nFold {fold_id}: {len(train_idx)} train, {len(val_idx)} val")

        best_loss, model_path, _ = train_one_fold(
            fold_id, train_idx, val_idx, dataset, best_params
        )

        fold_results.append((best_loss, model_path))

    # ✅ SELECT BEST MODEL
    best_fold = min(fold_results, key=lambda x: x[0])
    os.rename(best_fold[1], "best_model_cv.pth")

    print(f"\n✅ Best model: {best_fold[1]} → saved as best_model_cv.pth")

def analyze_graph_connectivity(dataset):
    total_residues = 0
    total_isolated_residues = 0
    pdbs_with_isolated = 0
    pdbs_not_complete = 0

    for data in dataset:
        n = data.num_nodes
        total_residues += n

        # Count node degrees
        degrees = torch.zeros(n)
        for i in data.edge_index[0]:
            degrees[i] += 1

        isolated = (degrees == 0).sum().item()
        total_isolated_residues += isolated

        if isolated > 0:
            pdbs_with_isolated += 1

        # Check if complete graph
        expected_edges = n * (n - 1)
        actual_edges = data.edge_index.size(1)

        if actual_edges < expected_edges:
            pdbs_not_complete += 1

    print("\n===== GRAPH ANALYSIS (Cutoff = 10 Å) =====")
    print(f"Total residues: {total_residues}")
    print(f"Total isolated residues: {total_isolated_residues}")
    print(f"Residues lost due to cutoff: {total_isolated_residues}")
    print(f"PDBs with isolated residues: {pdbs_with_isolated}")
    print(f"PDBs NOT complete graphs: {pdbs_not_complete}")

if __name__ == "__main__":
    folder = os.getcwd()

    # === Build dataset ===
    dataset = build_dataset_from_folder(folder)
    print(f"Built {len(dataset)} protein graphs.")

    # === NEW: Analyze graph connectivity (cutoff = 10 Å) ===
    analyze_graph_connectivity(dataset)

    # === Count flexible vs non-flexible residues ===
    flexible_count, non_flexible_count = count_flexible_residues(dataset)
    print(f"Total flexible residues: {flexible_count}")
    print(f"Total non-flexible residues: {non_flexible_count}")

    # === Hyperparameter optimization ===
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: train_and_evaluate(trial, dataset), n_trials=25)

    trial = study.best_trial
    print("Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # === Save best hyperparameters ===
    with open("best_hyperparameters.txt", "w") as f:
        f.write("Best trial hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest trial value (lowest validation loss): {trial.value:.4f}\n")

    # === Run 10-fold cross validation ===
    run_cross_validation(dataset, trial.params, n_splits=10)

