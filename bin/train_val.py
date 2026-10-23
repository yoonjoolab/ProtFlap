import os
import csv
import time
import glob
import shutil
import logging
import subprocess
import re

from copy import deepcopy

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import networkx as nx

from scipy.stats import pearsonr, spearmanr

# PyTorch
from torch.utils.data import DataLoader, random_split

# PyTorch Geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, SAGEConv, GATConv

# Sklearn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Bio / Features
from Bio import PDB
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from aaindex import aaindex1

from .gnn_model import NodeMLP_GCN
from .data_utils import load_rmsf_data
from .prediction_utils import get_predictions, compare_rmsf_and_predictions


# ============================================================
# GLOBAL SETTINGS
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("==============================================")
print("ProtFlap")
print("==============================================")
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CONSTANTS
# ============================================================

THREE_TO_ONE = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}

AMINO_ACIDS = 'GAVLMIWFYSTCPNQKRHDE'

AA_TO_INDEX = {
    aa: i
    for i, aa in enumerate(AMINO_ACIDS)
}

POLARITY_CATEGORIES = {
    'nonpolar': [
        'A', 'F', 'G', 'I', 'L',
        'M', 'P', 'V', 'W'
    ],
    'polar_uncharged': [
        'C', 'N', 'Q', 'S', 'T', 'Y'
    ],
    'polar_positive': [
        'H', 'K', 'R'
    ],
    'polar_negative': [
        'D', 'E'
    ]
}



# ============================================================
# CORRELATION FUNCTIONS
# ============================================================

def safe_pearson(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    mask = (
        np.isfinite(y_true)
        &
        np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    if (
        np.std(y_true) == 0
        or
        np.std(y_pred) == 0
    ):
        return np.nan

    return pearsonr(
        y_true,
        y_pred
    )[0]


def safe_spearman(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    mask = (
        np.isfinite(y_true)
        &
        np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    if (
        np.std(y_true) == 0
        or
        np.std(y_pred) == 0
    ):
        return np.nan

    return spearmanr(
        y_true,
        y_pred
    )[0]


# ============================================================
# FLEXIBLE RESIDUE COUNT
# ============================================================

def count_flexible_residues(
    dataset,
    threshold=0.0
):

    flexible_count = 0
    non_flexible_count = 0

    for data in dataset:

        rmsf_values = load_rmsf_data(
            data.pdb_name
        )

        if rmsf_values is None:
            continue

        for val in rmsf_values:

            if val > threshold:
                flexible_count += 1

            else:
                non_flexible_count += 1

    return (
        flexible_count,
        non_flexible_count
    )


# ============================================================
# FREESASA PARSER
# ============================================================

def parse_freesasa_output(output):

    lines = output.strip().split("\n")

    parsed_data = {}

    for line in lines:

        if (
            line.startswith("#")
            or
            not line.strip()
        ):
            continue

        parts = re.split(
            r"\s+",
            line.strip()
        )

        if len(parts) >= 5:

            try:

                residue_number = int(
                    re.sub(
                        r"\D",
                        "",
                        parts[2]
                    )
                )

                surface_area = float(
                    parts[5]
                    .replace(":", "")
                    .strip()
                )

                parsed_data[
                    residue_number
                ] = surface_area

            except ValueError:
                continue

    return parsed_data


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def one_hot_encode(
    residue_name
):

    index = AA_TO_INDEX.get(
        residue_name,
        -1
    )

    if index != -1:

        encoding = [0] * 20

        encoding[index] = 1

        return encoding

    return [0] * 20


# ============================================================
# POLARITY ENCODING
# ============================================================

def get_polarity_encoding(
    residue_name
):

    for category, residues in (
        POLARITY_CATEGORIES.items()
    ):

        if residue_name in residues:

            return [
                1 if category == cat else 0
                for cat in POLARITY_CATEGORIES.keys()
            ]

    return [
        0
        for _ in POLARITY_CATEGORIES
    ]


# ============================================================
# DSSP ENCODING
# ============================================================

def dssp_simplified_encode(
    dssp_value
):

    if dssp_value in [
        'H',
        'G',
        'I'
    ]:

        return [
            1,
            0,
            0
        ]

    elif dssp_value in [
        'E',
        'B'
    ]:

        return [
            0,
            1,
            0
        ]

    else:

        return [
            0,
            0,
            1
        ]


# ============================================================
# RESIDUE FEATURES
# ============================================================

def get_residue_features(
    residue,
    freesasa_value,
    atomic_energies,
    dssp_value
):

    residue_name = THREE_TO_ONE.get(
        residue.get_resname(),
        'X'
    )

    one_hot = one_hot_encode(
        residue_name
    )

    polarity_encoding = (
        get_polarity_encoding(
            residue_name
        )
    )

    hydrophobicity_values = (
        aaindex1[
            'KYTJ820101'
        ]['values']
    )

    size_values = (
        aaindex1[
            'FASG760101'
        ]['values']
    )

    charge_values = (
        aaindex1[
            'KLEP840101'
        ]['values']
    )

    hydrophobicity = (
        hydrophobicity_values.get(
            residue_name,
            0
        )
    )

    size = (
        size_values.get(
            residue_name,
            0
        )
    )

    charge = (
        charge_values.get(
            residue_name,
            0
        )
    )

    exposure = (
        1
        if freesasa_value >= 15
        else 0
    )

    dssp_encoding = (
        dssp_simplified_encode(
            dssp_value
        )
    )

    feature_vector = (
        one_hot
        +
        polarity_encoding
        +
        [
            hydrophobicity,
            size,
            charge,
            exposure
        ]
        +
        dssp_encoding
        +
        atomic_energies
    )

    return feature_vector


# ============================================================
# MOVE MISMATCHED FILES
# ============================================================

def move_mismatched_files(
    pdb_file,
    csv_file,
    destination_directory="mismatched_files"
):

    if not os.path.exists(
        destination_directory
    ):

        os.makedirs(
            destination_directory
        )

    pdb_name = os.path.basename(
        pdb_file
    )

    csv_name = os.path.basename(
        csv_file
    )

    pdb_destination = os.path.join(
        destination_directory,
        pdb_name
    )

    csv_destination = os.path.join(
        destination_directory,
        csv_name
    )

    try:

        shutil.move(
            pdb_file,
            pdb_destination
        )

    except Exception as e:

        logging.error(
            f"Error moving {pdb_name}: {e}"
        )

    try:

        shutil.move(
            csv_file,
            csv_destination
        )

    except Exception as e:

        logging.error(
            f"Error moving {csv_name}: {e}"
        )


# ============================================================
# LOAD PDB AND FEATURES
# ============================================================

def load_and_present_pdb(
    pdb_file
):

    pdb_name = os.path.splitext(
        os.path.basename(
            pdb_file
        )
    )[0]

    atomic_energy_csv = (
        f"{pdb_name}.csv"
    )

    if not os.path.exists(
        atomic_energy_csv
    ):

        logging.error(
            f"CSV not found for {pdb_name}"
        )

        return None

    parser = PDB.PDBParser(
        QUIET=True
    )

    try:

        structure = (
            parser.get_structure(
                "structure",
                pdb_file
            )
        )

    except Exception as e:

        logging.error(
            f"PDB parsing error "
            f"{pdb_name}: {e}"
        )

        return None

    try:

        atomic_energy_data = (
            pd.read_csv(
                atomic_energy_csv
            )
        )

    except Exception as e:

        logging.error(
            f"Error reading energy CSV "
            f"{pdb_name}: {e}"
        )

        return None

    # --------------------------------------------------------
    # FreeSASA
    # --------------------------------------------------------

    freesasa_command = (
        f"freesasa "
        f"--foreach-residue "
        f"--no-log "
        f"{pdb_file}"
    )

    try:

        freesasa_result = subprocess.run(
            freesasa_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )

        freesasa_dict = (
            parse_freesasa_output(
                freesasa_result.stdout
            )
        )

    except Exception as e:

        logging.error(
            f"FreeSASA failed for "
            f"{pdb_name}: {e}"
        )

        return None

    # --------------------------------------------------------
    # Residue numbers
    # --------------------------------------------------------

    def clean_and_convert_residue(
        residue
    ):

        residue = (
            str(residue)
            .strip()
            .replace(',', '')
        )

        try:

            return int(residue)

        except ValueError:

            return None

    atomic_energy_data[
        "Residue"
    ] = atomic_energy_data[
        "Residue"
    ].apply(
        clean_and_convert_residue
    )

    atomic_energy_data = (
        atomic_energy_data.dropna(
            subset=["Residue"]
        )
    )

    # 28 energy terms
    atomic_energy_dict = dict(
        zip(
            atomic_energy_data[
                "Residue"
            ],
            atomic_energy_data.iloc[
                :,
                1:29
            ].values.tolist()
        )
    )

    # --------------------------------------------------------
    # DSSP
    # --------------------------------------------------------

    try:

        dssp_dict = (
            dssp_dict_from_pdb_file(
                pdb_file
            )[0]
        )

    except Exception as e:

        logging.error(
            f"DSSP failed for "
            f"{pdb_name}: {e}"
        )

        return None

    # --------------------------------------------------------
    # Feature construction
    # --------------------------------------------------------

    all_node_features = []

    protein_graphs = {}

    node_features_dict = {}

    residue_counter = 0

    pdb_residue_count = 0

    for model in structure:

        for chain in model:

            residues_list = [
                residue
                for residue in chain
                if PDB.is_aa(residue)
            ]

            chain_node_features = []

            chain_node_features_dict = {}

            protein_graph = (
                create_protein_graph(
                    structure,
                    chain.get_id()
                )
            )

            protein_graphs[
                chain.get_id()
            ] = protein_graph

            for residue in residues_list:

                pdb_residue_count += 1

                try:

                    residue_id = (
                        residue.get_id()
                    )

                    residue_number = (
                        residue_id[1]
                    )

                    freesasa_value = (
                        freesasa_dict.get(
                            residue_number,
                            0
                        )
                    )

                    atomic_energies = (
                        atomic_energy_dict.get(
                            residue_number,
                            [0] * 28
                        )
                    )

                    dssp_value = (
                        dssp_dict.get(
                            (
                                chain.get_id(),
                                residue_id
                            ),
                            '-'
                        )
                    )

                    features = (
                        get_residue_features(
                            residue,
                            freesasa_value,
                            atomic_energies,
                            dssp_value
                        )
                    )

                    chain_node_features.append(
                        features
                    )

                    chain_node_features_dict[
                        residue_counter
                    ] = features

                    residue_counter += 1

                except Exception as e:

                    logging.warning(
                        f"Error processing residue: {e}"
                    )

                    continue

            all_node_features.extend(
                chain_node_features
            )

            node_features_dict[
                chain.get_id()
            ] = chain_node_features_dict

    # --------------------------------------------------------
    # Residue count validation
    # --------------------------------------------------------

    csv_residue_count = (
        len(atomic_energy_data)
    )

    if not all_node_features:

        logging.error(
            f"No valid residues in {pdb_name}"
        )

        return None

    if pdb_residue_count != csv_residue_count:

        logging.warning(
            f"Mismatch {pdb_name}: "
            f"PDB={pdb_residue_count}, "
            f"CSV={csv_residue_count}"
        )

        return None

    node_features_tensor = torch.tensor(
        all_node_features,
        dtype=torch.float
    )

    print(
        f"Processed PDB: {pdb_name}, "
        f"nodes={node_features_tensor.shape[0]}, "
        f"features={node_features_tensor.shape[1]}"
    )

    return (
        node_features_tensor,
        protein_graphs,
        node_features_dict
    )


# ============================================================
# CREATE PROTEIN GRAPH
# ============================================================

def create_protein_graph(
    structure,
    chain_id,
    max_distance=15.0
):

    graph = nx.Graph()

    chain = structure[0][
        chain_id
    ]

    residues = [
        residue
        for residue in chain
        if PDB.is_aa(residue)
    ]

    for idx, residue in enumerate(
        residues
    ):

        graph.add_node(
            idx,
            residue=residue
        )

    for i, res1 in enumerate(
        residues
    ):

        for j, res2 in enumerate(
            residues[i + 1:],
            start=i + 1
        ):

            if (
                'CA' not in res1
                or
                'CA' not in res2
            ):
                continue

            distance = np.linalg.norm(
                res1['CA'].coord
                -
                res2['CA'].coord
            )

            if distance < max_distance:

                graph.add_edge(
                    i,
                    j,
                    distance=distance
                )

    return graph


# ============================================================
# BUILD DATASET
#
# Runtime is measured here so that the reviewer runtime
# corresponds to the actual preprocessing pipeline.
# ============================================================

def build_dataset_from_folder(
    folder
):

    data_list = []

    runtime_records = []

    pdb_files = sorted([
        filename
        for filename in os.listdir(folder)
        if filename.endswith(".pdb")
    ])

    print(
        f"\nFound {len(pdb_files)} PDB files."
    )

    for counter, filename in enumerate(
        pdb_files,
        start=1
    ):

        pdb_path = os.path.join(
            folder,
            filename
        )

        pdb_name = os.path.splitext(
            filename
        )[0]

        print(
            f"\n[{counter}/{len(pdb_files)}] "
            f"{pdb_name}"
        )

        # ----------------------------------------------------
        # START PREPROCESSING TIMER
        # ----------------------------------------------------

        start_time = time.perf_counter()

        result = (
            load_and_present_pdb(
                pdb_path
            )
        )

        elapsed_time = (
            time.perf_counter()
            -
            start_time
        )

        # ----------------------------------------------------
        # Runtime record
        # ----------------------------------------------------

        if result is not None:

            runtime_records.append({

                "PDB_Name":
                    pdb_name,

                "Preprocessing_seconds":
                    elapsed_time

            })

        if result is None:

            continue

        (
            node_features_tensor,
            protein_graphs,
            node_features_dict
        ) = result

        # ----------------------------------------------------
        # RMSF CSV
        # ----------------------------------------------------

        csv_file = os.path.join(
            folder,
            f"{pdb_name}.csv"
        )

        if not os.path.exists(
            csv_file
        ):

            print(
                f"Skipping {pdb_name}: "
                f"CSV not found"
            )

            continue

        df = pd.read_csv(
            csv_file
        )

        if "rmsf_norm" not in df.columns:

            print(
                f"Skipping {pdb_name}: "
                f"rmsf_norm missing"
            )

            continue

        rmsf = (
            df["rmsf_norm"]
            .values
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Build graph
        # ----------------------------------------------------

        all_x = []

        src_list = []

        dst_list = []

        edge_weights = []

        node_offset = 0

        for chain_id, G in (
            protein_graphs.items()
        ):

            n = G.number_of_nodes()

            if n == 0:
                continue

            x_chain = np.array([
                node_features_dict[
                    chain_id
                ][i]
                for i in range(n)
            ])

            # Feature scaling
            scaler_x = StandardScaler()

            x_chain = (
                scaler_x.fit_transform(
                    x_chain
                )
            )

            all_x.append(
                x_chain
            )

            for u, v, attrs in (
                G.edges(data=True)
            ):

                dist = attrs.get(
                    "distance",
                    15.0
                )

                weight = (
                    1.0
                    /
                    (dist + 1e-6)
                )

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

            print(
                f"Skipping {pdb_name}: "
                f"no node features"
            )

            continue

        x = np.vstack(
            all_x
        )

        total_nodes = (
            x.shape[0]
        )

        # ----------------------------------------------------
        # Exact residue matching
        # ----------------------------------------------------

        if total_nodes != len(rmsf):

            print(
                f"Skipping {pdb_name}: "
                f"graph={total_nodes}, "
                f"RMSF={len(rmsf)}"
            )

            continue

        x = torch.tensor(
            x,
            dtype=torch.float
        )

        y = torch.tensor(
            rmsf,
            dtype=torch.float
        )

        # ----------------------------------------------------
        # Edges
        # ----------------------------------------------------

        if len(src_list) == 0:

            edge_index = (
                torch.arange(
                    total_nodes,
                    dtype=torch.long
                ).repeat(
                    2,
                    1
                )
            )

            edge_attr = torch.ones(
                edge_index.size(1),
                dtype=torch.float
            )

        else:

            edge_index = torch.tensor(
                [
                    src_list,
                    dst_list
                ],
                dtype=torch.long
            )

            edge_attr = torch.tensor(
                edge_weights,
                dtype=torch.float
            )

        # ----------------------------------------------------
        # PyG Data
        # ----------------------------------------------------

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y
        )

        data.pdb_name = pdb_name

        data.residue_indices = (
            torch.arange(
                total_nodes,
                dtype=torch.long
            )
        )

        data_list.append(
            data
        )

        print(
            f"Added {pdb_name}: "
            f"{total_nodes} residues | "
            f"preprocessing "
            f"{elapsed_time:.3f} s"
        )

    # --------------------------------------------------------
    # Save preprocessing runtime
    # --------------------------------------------------------

    runtime_df = pd.DataFrame(
        runtime_records
    )

    runtime_file = os.path.join(
        ANALYSIS_DIR,
        "preprocessing_runtime_per_structure.csv"
    )

    runtime_df.to_csv(
        runtime_file,
        index=False
    )

    if len(runtime_df) > 0:

        mean_time = (
            runtime_df[
                "Preprocessing_seconds"
            ].mean()
        )

        median_time = (
            runtime_df[
                "Preprocessing_seconds"
            ].median()
        )

        print("\n==============================================")
        print("PREPROCESSING RUNTIME")
        print("==============================================")

        print(
            f"Structures: "
            f"{len(runtime_df)}"
        )

        print(
            f"Mean: "
            f"{mean_time:.4f} s"
        )

        print(
            f"Median: "
            f"{median_time:.4f} s"
        )

        if mean_time > 0:

            print(
                f"Structures/hour: "
                f"{3600 / mean_time:.2f}"
            )

    print(
        f"\nRuntime saved to:\n"
        f"{runtime_file}"
    )

    return data_list


# ============================================================
# OPTUNA TRAINING
# ============================================================

def train_and_evaluate(
    trial,
    data_list
):

    lr = trial.suggest_categorical(
        "learning_rate",
        [
            1e-2,
            1e-3,
            1e-4
        ]
    )

    dropout = trial.suggest_float(
        "dropout",
        0.001,
        0.3
    )

    weight_decay = trial.suggest_categorical(
        "weight_decay",
        [
            1e-4,
            1e-3,
            1e-2
        ]
    )

    batch_norm = trial.suggest_categorical(
        "batch_norm",
        [
            True,
            False
        ]
    )

    residual = trial.suggest_categorical(
        "residual",
        [
            True,
            False
        ]
    )

    activation = trial.suggest_categorical(
        "activation",
        [
            "ReLU",
            "ELU"
        ]
    )

    use_bias = trial.suggest_categorical(
        "use_bias",
        [
            True,
            False
        ]
    )

    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [
            64,
            128,
            256
        ]
    )

    num_gcn_layers = trial.suggest_int(
        "num_gcn_layers",
        2,
        6
    )

    n = len(
        data_list
    )

    n_train = int(
        0.9 * n
    )

    train_ds, val_ds = random_split(
        data_list,
        [
            n_train,
            n - n_train
        ]
    )

    train_loader = (
        GeometricDataLoader(
            train_ds,
            batch_size=16,
            shuffle=True
        )
    )

    val_loader = (
        GeometricDataLoader(
            val_ds,
            batch_size=16,
            shuffle=False
        )
    )

    model = NodeMLP_GCN(
        data_list[0].x.size(1),
        hidden_dim=hidden_dim,
        num_gcn_layers=num_gcn_layers,
        dropout=dropout,
        use_residual=residual,
        use_batch_norm=batch_norm,
        activation=activation,
        use_bias=use_bias
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = (
        torch.nn.SmoothL1Loss()
    )

    best_val_loss = float(
        "inf"
    )

    patience = 20

    patience_counter = 0

    for epoch in range(100):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        total_loss = 0

        total_nodes = 0

        for batch in train_loader:

            batch = batch.to(
                device
            )

            optimizer.zero_grad()

            preds, _ = model(
                batch.x,
                batch.edge_index
            )

            loss = criterion(
                preds,
                batch.y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                *
                batch.num_nodes
            )

            total_nodes += (
                batch.num_nodes
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        total_val_loss = 0

        total_val_nodes = 0

        with torch.no_grad():

            for batch in val_loader:

                batch = batch.to(
                    device
                )

                preds, _ = model(
                    batch.x,
                    batch.edge_index
                )

                loss = criterion(
                    preds,
                    batch.y
                )

                total_val_loss += (
                    loss.item()
                    *
                    batch.num_nodes
                )

                total_val_nodes += (
                    batch.num_nodes
                )

        val_loss = (
            total_val_loss
            /
            total_val_nodes
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            break

    return best_val_loss


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    fold_id,
    train_idx,
    val_idx,
    dataset,
    best_params,
    device="cpu",
    epochs=100,
    batch_size=16,
    patience=20
):

    print("\n")
    print("==============================================")
    print(f"FOLD {fold_id}")
    print("==============================================")

    train_ds = [
        dataset[i]
        for i in train_idx
    ]

    val_ds = [
        dataset[i]
        for i in val_idx
    ]

    train_loader = (
        GeometricDataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True
        )
    )

    val_loader = (
        GeometricDataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False
        )
    )

    in_node_feats = (
        dataset[0].x.size(1)
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = NodeMLP_GCN(
        in_node_feats,
        hidden_dim=best_params[
            "hidden_dim"
        ],
        num_gcn_layers=best_params[
            "num_gcn_layers"
        ],
        dropout=best_params[
            "dropout"
        ],
        use_residual=best_params[
            "residual"
        ],
        use_batch_norm=best_params[
            "batch_norm"
        ],
        activation=best_params[
            "activation"
        ],
        use_bias=best_params[
            "use_bias"
        ]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params[
            "learning_rate"
        ],
        weight_decay=best_params[
            "weight_decay"
        ]
    )

    criterion = (
        torch.nn.SmoothL1Loss()
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.7,
            patience=10
        )
    )

    best_val_loss = float(
        "inf"
    )

    best_model_state = None

    patience_counter = 0

    train_losses = []

    val_losses = []

    best_epoch_preds = None

    best_epoch_targets = None

    best_epoch_pdb_names = None

    best_epoch_residue_indices = None

    best_epoch_metrics = None

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        total_train_loss = 0.0

        total_train_nodes = 0

        for batch in train_loader:

            batch = batch.to(
                device
            )

            optimizer.zero_grad()

            preds, _ = model(
                batch.x,
                batch.edge_index
            )

            loss = criterion(
                preds,
                batch.y
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
                *
                batch.num_nodes
            )

            total_train_nodes += (
                batch.num_nodes
            )

        train_loss = (
            total_train_loss
            /
            total_train_nodes
        )

        train_losses.append(
            train_loss
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        total_val_loss = 0.0

        total_val_nodes = 0

        all_preds = []

        all_targets = []

        pdb_names = []

        residue_indices = []

        with torch.no_grad():

            for batch in val_loader:

                batch = batch.to(
                    device
                )

                preds, _ = model(
                    batch.x,
                    batch.edge_index
                )

                loss = criterion(
                    preds,
                    batch.y
                )

                total_val_loss += (
                    loss.item()
                    *
                    batch.num_nodes
                )

                total_val_nodes += (
                    batch.num_nodes
                )

                batch_preds = (
                    preds
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )

                batch_targets = (
                    batch.y
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )

                all_preds.extend(
                    batch_preds
                )

                all_targets.extend(
                    batch_targets
                )

                ptr = (
                    batch.ptr
                    .cpu()
                    .numpy()
                )

                batch_residue_indices = (
                    batch.residue_indices
                    .cpu()
                    .numpy()
                )

                batch_pdb_names = (
                    batch.pdb_name
                )

                if not isinstance(
                    batch_pdb_names,
                    (list, tuple, np.ndarray)
                ):

                    batch_pdb_names = [
                        batch_pdb_names
                    ]

                for i in range(
                    len(batch_pdb_names)
                ):

                    start = ptr[i]

                    end = ptr[i + 1]

                    n_nodes_graph = (
                        end - start
                    )

                    pdb_names.extend(
                        [
                            batch_pdb_names[i]
                        ]
                        *
                        n_nodes_graph
                    )

                    residue_indices.extend(
                        batch_residue_indices[
                            start:end
                        ].tolist()
                    )

        val_loss = (
            total_val_loss
            /
            total_val_nodes
        )

        val_losses.append(
            val_loss
        )

        # ====================================================
        # METRICS
        # ====================================================

        all_preds_np = np.array(
            all_preds
        )

        all_targets_np = np.array(
            all_targets
        )

        rmse = np.sqrt(
            mean_squared_error(
                all_targets_np,
                all_preds_np
            )
        )

        mae = mean_absolute_error(
            all_targets_np,
            all_preds_np
        )

        r2 = r2_score(
            all_targets_np,
            all_preds_np
        )

        pearson = safe_pearson(
            all_targets_np,
            all_preds_np
        )

        spearman = safe_spearman(
            all_targets_np,
            all_preds_np
        )

        scheduler.step(
            val_loss
        )

        print(
            f"[Fold {fold_id}] "
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"RMSE: {rmse:.4f} | "
            f"MAE: {mae:.4f} | "
            f"R2: {r2:.4f} | "
            f"Pearson: {pearson:.4f} | "
            f"Spearman: {spearman:.4f}"
        )

        # ====================================================
        # SAVE BEST
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            # IMPORTANT:
            # deepcopy prevents the state from changing
            best_model_state = deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            best_epoch_preds = (
                all_preds_np.copy()
            )

            best_epoch_targets = (
                all_targets_np.copy()
            )

            best_epoch_pdb_names = (
                pdb_names.copy()
            )

            best_epoch_residue_indices = (
                residue_indices.copy()
            )

            best_epoch_metrics = {
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Pearson": pearson,
                "Spearman": spearman
            }

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(
                f"[Fold {fold_id}] "
                f"Early stopping at "
                f"epoch {epoch}"
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    if best_model_state is None:

        raise ValueError(
            f"No best model saved "
            f"for fold {fold_id}"
        )

    model.load_state_dict(
        best_model_state
    )

    model_path = (
        f"best_model_fold{fold_id}.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    # ========================================================
    # SAVE PER-RESIDUE PREDICTIONS
    # ========================================================

    df = compare_rmsf_and_predictions(
        best_epoch_pdb_names,
        best_epoch_residue_indices,
        best_epoch_targets,
        best_epoch_preds
    )

    prediction_file = (
        f"fold_{fold_id}_predictions.csv"
    )

    df.to_csv(
        prediction_file,
        index=False
    )

    # ========================================================
    # SAVE METRICS TXT
    # ========================================================

    metrics_file = (
        f"fold_{fold_id}_metrics.txt"
    )

    with open(
        metrics_file,
        "w"
    ) as f:

        f.write(
            f"RMSE: "
            f"{best_epoch_metrics['RMSE']:.4f}\n"
        )

        f.write(
            f"MAE: "
            f"{best_epoch_metrics['MAE']:.4f}\n"
        )

        f.write(
            f"R2: "
            f"{best_epoch_metrics['R2']:.4f}\n"
        )

        f.write(
            f"Pearson: "
            f"{best_epoch_metrics['Pearson']:.4f}\n"
        )

        f.write(
            f"Spearman: "
            f"{best_epoch_metrics['Spearman']:.4f}\n"
        )

    # ========================================================
    # LOSS CURVE
    # ========================================================

    plt.figure()

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("SmoothL1 Loss")

    plt.title(
        f"Loss Curve Fold {fold_id}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"loss_curve_fold{fold_id}.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # SCATTER
    # ========================================================

    plt.figure()

    plt.scatter(
        best_epoch_targets,
        best_epoch_preds,
        alpha=0.5
    )

    plt.xlabel(
        "Target RMSF_norm"
    )

    plt.ylabel(
        "Predicted RMSF_norm"
    )

    plt.title(
        f"Fold {fold_id}"
    )

    plt.tight_layout()

    plt.savefig(
        f"scatter_fold{fold_id}.png",
        dpi=300
    )

    plt.close()

    print(
        f"\n[Fold {fold_id}] "
        f"Best metrics:"
    )

    for key, value in (
        best_epoch_metrics.items()
    ):

        print(
            f"  {key}: {value:.4f}"
        )

    return (
        best_val_loss,
        model_path,
        best_epoch_metrics
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    dataset,
    best_params,
    n_splits=10
):

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    fold_results = []

    for fold_id, (
        train_idx,
        val_idx
    ) in enumerate(
        kf.split(dataset),
        1
    ):

        print(
            f"\nFold {fold_id}: "
            f"{len(train_idx)} train, "
            f"{len(val_idx)} validation"
        )

        (
            best_loss,
            model_path,
            metrics
        ) = train_one_fold(
            fold_id,
            train_idx,
            val_idx,
            dataset,
            best_params,
            device=device
        )

        fold_results.append({

            "Fold":
                fold_id,

            "Best_Val_Loss":
                best_loss,

            "Model":
                model_path,

            "RMSE":
                metrics["RMSE"],

            "MAE":
                metrics["MAE"],

            "R2":
                metrics["R2"],

            "Pearson":
                metrics["Pearson"],

            "Spearman":
                metrics["Spearman"]

        })

    # --------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------

    fold_df = pd.DataFrame(
        fold_results
    )

    fold_df.to_csv(
        "cross_validation_summary.csv",
        index=False
    )

    print("\n==============================================")
    print("10-FOLD CROSS VALIDATION SUMMARY")
    print("==============================================")

    print(
        fold_df[
            [
                "Fold",
                "RMSE",
                "MAE",
                "R2",
                "Pearson",
                "Spearman"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nMean:")
    print(
        fold_df[
            [
                "RMSE",
                "MAE",
                "R2",
                "Pearson",
                "Spearman"
            ]
        ].mean()
    )

    print("\nSD:")
    print(
        fold_df[
            [
                "RMSE",
                "MAE",
                "R2",
                "Pearson",
                "Spearman"
            ]
        ].std()
    )

    # --------------------------------------------------------
    # Select best fold
    # --------------------------------------------------------

    best_fold = min(
        fold_results,
        key=lambda x: x[
            "Best_Val_Loss"
        ]
    )

    if os.path.exists(
        "best_model_cv.pth"
    ):

        os.remove(
            "best_model_cv.pth"
        )

    os.rename(
        best_fold["Model"],
        "best_model_cv.pth"
    )

    print(
        f"\nBest model: "
        f"{best_fold['Model']} "
        f"-> best_model_cv.pth"
    )

    return fold_df


# ============================================================
# GRAPH CONNECTIVITY
# ============================================================

def analyze_graph_connectivity(
    dataset
):

    total_residues = 0

    total_isolated_residues = 0

    pdbs_with_isolated = 0

    pdbs_not_complete = 0

    for data in dataset:

        n = data.num_nodes

        total_residues += n

        degrees = torch.zeros(
            n
        )

        for i in data.edge_index[0]:

            degrees[i] += 1

        isolated = (
            degrees == 0
        ).sum().item()

        total_isolated_residues += (
            isolated
        )

        if isolated > 0:

            pdbs_with_isolated += 1

        expected_edges = (
            n * (n - 1)
        )

        actual_edges = (
            data.edge_index.size(1)
        )

        if actual_edges < expected_edges:

            pdbs_not_complete += 1

    print(
        "\n===== GRAPH ANALYSIS "
        "(Cutoff = 15 Å) ====="
    )

    print(
        f"Total residues: "
        f"{total_residues}"
    )

    print(
        f"Total isolated residues: "
        f"{total_isolated_residues}"
    )

    print(
        f"PDBs with isolated residues: "
        f"{pdbs_with_isolated}"
    )

    print(
        f"PDBs NOT complete graphs: "
        f"{pdbs_not_complete}"
    )


# ============================================================
# PER-FOLD ANALYSIS FROM CSV
#
# This uses your existing:
#
# fold_1_predictions.csv
# fold_2_predictions.csv
# ...
#
# ============================================================

def calculate_fold_metrics_from_csv():

    files = sorted(
        glob.glob(
            "fold_*_predictions.csv"
        )
    )

    if len(files) == 0:

        print(
            "No fold prediction CSVs found."
        )

        return None

    results = []

    print("\n")
    print("==============================================")
    print("PER-FOLD DISTRIBUTIONAL ANALYSIS")
    print("==============================================")

    for file in files:

        match = re.search(
            r"fold_(\d+)_predictions\.csv",
            file
        )

        if match:

            fold_id = int(
                match.group(1)
            )

        else:

            fold_id = file

        df = pd.read_csv(
            file
        )

        # ----------------------------------------------------
        # Identify columns
        # ----------------------------------------------------

        target_col = None

        pred_col = None

        for col in [
            "Target_RMSF",
            "Target",
            "rmsf_norm",
            "RMSF",
            "Actual_RMSF"
        ]:

            if col in df.columns:

                target_col = col

                break

        for col in [
            "Predicted_RMSF",
            "Prediction",
            "Predicted"
        ]:

            if col in df.columns:

                pred_col = col

                break

        if (
            target_col is None
            or
            pred_col is None
        ):

            print(
                f"Skipping {file}"
            )

            print(
                df.columns.tolist()
            )

            continue

        y_true = (
            df[target_col]
            .astype(float)
            .values
        )

        y_pred = (
            df[pred_col]
            .astype(float)
            .values
        )

        mask = (
            np.isfinite(y_true)
            &
            np.isfinite(y_pred)
        )

        y_true = y_true[mask]

        y_pred = y_pred[mask]

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        pearson = safe_pearson(
            y_true,
            y_pred
        )

        spearman = safe_spearman(
            y_true,
            y_pred
        )

        results.append({

            "Fold":
                fold_id,

            "N_residues":
                len(y_true),

            "RMSE":
                rmse,

            "MAE":
                mae,

            "R2":
                r2,

            "Pearson":
                pearson,

            "Spearman":
                spearman
        })

    results_df = pd.DataFrame(
        results
    )

    output = os.path.join(
        ANALYSIS_DIR,
        "per_fold_distribution_metrics.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nMean:")
    print(
        results_df[
            [
                "RMSE",
                "MAE",
                "R2",
                "Pearson",
                "Spearman"
            ]
        ].mean()
    )

    print("\nSD:")
    print(
        results_df[
            [
                "RMSE",
                "MAE",
                "R2",
                "Pearson",
                "Spearman"
            ]
        ].std()
    )

    print(
        f"\nSaved:\n{output}"
    )

    return results_df


# ============================================================
# PER-CLASS ANALYSIS
#
# rmsf_norm >= 0 = Flexible
# rmsf_norm < 0  = Non-flexible
# ============================================================

def calculate_per_class_metrics():

    files = sorted(
        glob.glob(
            "fold_*_predictions.csv"
        )
    )

    all_data = []

    for file in files:

        df = pd.read_csv(
            file
        )

        target_col = None

        pred_col = None

        for col in [
            "Target_RMSF",
            "Target",
            "rmsf_norm",
            "RMSF",
            "Actual_RMSF"
        ]:

            if col in df.columns:

                target_col = col

                break

        for col in [
            "Predicted_RMSF",
            "Prediction",
            "Predicted"
        ]:

            if col in df.columns:

                pred_col = col

                break

        if (
            target_col is None
            or
            pred_col is None
        ):

            continue

        temp = df[
            [target_col, pred_col]
        ].copy()

        temp.columns = [
            "Target",
            "Prediction"
        ]

        temp["Fold"] = os.path.basename(
            file
        )

        all_data.append(
            temp
        )

    if len(all_data) == 0:

        print(
            "No prediction data."
        )

        return None

    df_all = pd.concat(
        all_data,
        ignore_index=True
    )

    df_all["Class"] = np.where(
        df_all["Target"] >= 0,
        "Flexible",
        "Non-flexible"
    )

    results = []

    print("\n")
    print("==============================================")
    print("PER-CLASS ERROR ANALYSIS")
    print("==============================================")

    for class_name in [
        "Flexible",
        "Non-flexible"
    ]:

        sub = df_all[
            df_all["Class"] == class_name
        ]

        y_true = sub[
            "Target"
        ].values

        y_pred = sub[
            "Prediction"
        ].values

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        pearson = safe_pearson(
            y_true,
            y_pred
        )

        spearman = safe_spearman(
            y_true,
            y_pred
        )

        results.append({

            "Class":
                class_name,

            "N":
                len(sub),

            "RMSE":
                rmse,

            "MAE":
                mae,

            "R2":
                r2,

            "Pearson":
                pearson,

            "Spearman":
                spearman
        })

        print(
            f"\n{class_name}"
        )

        print(
            f"N        = {len(sub)}"
        )

        print(
            f"RMSE     = {rmse:.4f}"
        )

        print(
            f"MAE      = {mae:.4f}"
        )

        print(
            f"R2       = {r2:.4f}"
        )

        print(
            f"Pearson  = {pearson:.4f}"
        )

        print(
            f"Spearman = {spearman:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    output = os.path.join(
        ANALYSIS_DIR,
        "per_class_metrics.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved:\n{output}"
    )

    return results_df


# ============================================================
# PER-PROTEIN ANALYSIS
# ============================================================

def calculate_per_protein_metrics():

    files = sorted(
        glob.glob(
            "fold_*_predictions.csv"
        )
    )

    results = []

    for file in files:

        df = pd.read_csv(
            file
        )

        if "PDB_Name" not in df.columns:

            print(
                f"PDB_Name missing: {file}"
            )

            continue

        target_col = None

        pred_col = None

        for col in [
            "Target_RMSF",
            "Target",
            "rmsf_norm",
            "RMSF",
            "Actual_RMSF"
        ]:

            if col in df.columns:

                target_col = col

                break

        for col in [
            "Predicted_RMSF",
            "Prediction",
            "Predicted"
        ]:

            if col in df.columns:

                pred_col = col

                break

        if (
            target_col is None
            or
            pred_col is None
        ):

            continue

        for pdb_name, group in (
            df.groupby(
                "PDB_Name"
            )
        ):

            y_true = (
                group[target_col]
                .astype(float)
                .values
            )

            y_pred = (
                group[pred_col]
                .astype(float)
                .values
            )

            mask = (
                np.isfinite(y_true)
                &
                np.isfinite(y_pred)
            )

            y_true = y_true[mask]

            y_pred = y_pred[mask]

            if len(y_true) < 2:

                continue

            errors = (
                y_pred - y_true
            )

            abs_errors = np.abs(
                errors
            )

            results.append({

                "PDB_Name":
                    pdb_name,

                "Fold":
                    os.path.basename(
                        file
                    ),

                "N_residues":
                    len(y_true),

                "Target_mean":
                    np.mean(y_true),

                "Target_std":
                    np.std(y_true),

                "Prediction_mean":
                    np.mean(y_pred),

                "Prediction_std":
                    np.std(y_pred),

                "Mean_Error":
                    np.mean(errors),

                "Mean_Absolute_Error":
                    np.mean(abs_errors),

                "Max_Absolute_Error":
                    np.max(abs_errors),

                "RMSE":
                    np.sqrt(
                        mean_squared_error(
                            y_true,
                            y_pred
                        )
                    ),

                "MAE":
                    mean_absolute_error(
                        y_true,
                        y_pred
                    ),

                "R2":
                    r2_score(
                        y_true,
                        y_pred
                    ),

                "Pearson":
                    safe_pearson(
                        y_true,
                        y_pred
                    ),

                "Spearman":
                    safe_spearman(
                        y_true,
                        y_pred
                    )
            })

    results_df = pd.DataFrame(
        results
    )

    output = os.path.join(
        ANALYSIS_DIR,
        "per_protein_metrics.csv"
    )

    results_df.to_csv(
        output,
        index=False
    )

    print("\n")
    print("==============================================")
    print("PER-PROTEIN DISTRIBUTIONAL ANALYSIS")
    print("==============================================")

    print(
        f"Proteins analyzed: "
        f"{len(results_df)}"
    )

    if len(results_df) > 0:

        metrics = [
            "RMSE",
            "MAE",
            "R2",
            "Pearson",
            "Spearman"
        ]

        print("\nMean:")
        print(
            results_df[
                metrics
            ].mean()
        )

        print("\nMedian:")
        print(
            results_df[
                metrics
            ].median()
        )

        print("\nSD:")
        print(
            results_df[
                metrics
            ].std()
        )

        print("\n25th percentile:")
        print(
            results_df[
                metrics
            ].quantile(
                0.25
            )
        )

        print("\n75th percentile:")
        print(
            results_df[
                metrics
            ].quantile(
                0.75
            )
        )

    print(
        f"\nSaved:\n{output}"
    )

    return results_df


# ============================================================
# FAILURE ANALYSIS
# ============================================================

def identify_failure_cases(
    top_n=20
):

    file = os.path.join(
        ANALYSIS_DIR,
        "per_protein_metrics.csv"
    )

    df = pd.read_csv(
        file
    )

    # --------------------------------------------------------
    # Worst RMSE
    # --------------------------------------------------------

    worst_rmse = (
        df.sort_values(
            "RMSE",
            ascending=False
        )
        .head(top_n)
    )

    worst_rmse.to_csv(
        os.path.join(
            ANALYSIS_DIR,
            "failure_cases_top_RMSE.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Worst MAE
    # --------------------------------------------------------

    worst_mae = (
        df.sort_values(
            "MAE",
            ascending=False
        )
        .head(top_n)
    )

    worst_mae.to_csv(
        os.path.join(
            ANALYSIS_DIR,
            "failure_cases_top_MAE.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Lowest Pearson
    # --------------------------------------------------------

    worst_pearson = (
        df.sort_values(
            "Pearson",
            ascending=True
        )
        .head(top_n)
    )

    worst_pearson.to_csv(
        os.path.join(
            ANALYSIS_DIR,
            "failure_cases_lowest_Pearson.csv"
        ),
        index=False
    )

    print("\n")
    print("==============================================")
    print("FAILURE CASE ANALYSIS")
    print("==============================================")

    print("\nWorst RMSE:")
    print(
        worst_rmse[
            [
                "PDB_Name",
                "N_residues",
                "RMSE",
                "MAE",
                "R2",
                "Pearson",
                "Spearman"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nWorst Pearson:")
    print(
        worst_pearson[
            [
                "PDB_Name",
                "N_residues",
                "RMSE",
                "MAE",
                "R2",
                "Pearson",
                "Spearman"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Extract residue-level failure data
    # --------------------------------------------------------

    prediction_files = glob.glob(
        "fold_*_predictions.csv"
    )

    failure_names = set(
        worst_rmse[
            "PDB_Name"
        ].astype(str)
    )

    failure_rows = []

    for file in prediction_files:

        df_pred = pd.read_csv(
            file
        )

        if "PDB_Name" not in df_pred:

            continue

        if (
            "Target_RMSF"
            not in df_pred.columns
            or
            "Predicted_RMSF"
            not in df_pred.columns
        ):

            continue

        selected = df_pred[
            df_pred["PDB_Name"]
            .astype(str)
            .isin(
                failure_names
            )
        ].copy()

        if len(selected) == 0:

            continue

        selected[
            "Absolute_Error"
        ] = np.abs(
            selected["Predicted_RMSF"]
            -
            selected["Target_RMSF"]
        )

        selected[
            "Error"
        ] = (
            selected["Predicted_RMSF"]
            -
            selected["Target_RMSF"]
        )

        selected[
            "Fold"
        ] = file

        failure_rows.append(
            selected
        )

    if len(failure_rows) > 0:

        failure_residue_df = (
            pd.concat(
                failure_rows,
                ignore_index=True
            )
        )

        failure_residue_df.to_csv(
            os.path.join(
                ANALYSIS_DIR,
                "failure_cases_residue_level.csv"
            ),
            index=False
        )


# ============================================================
# FAILURE DISTRIBUTION PLOTS
# ============================================================

def plot_error_distributions():

    file = os.path.join(
        ANALYSIS_DIR,
        "per_protein_metrics.csv"
    )

    if not os.path.exists(
        file
    ):

        return

    df = pd.read_csv(
        file
    )

    # --------------------------------------------------------
    # RMSE distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    sns.histplot(
        df["RMSE"],
        kde=True
    )

    plt.xlabel(
        "Per-protein RMSE"
    )

    plt.ylabel(
        "Number of proteins"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ANALYSIS_DIR,
            "per_protein_RMSE_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Pearson distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    sns.histplot(
        df["Pearson"].dropna(),
        kde=True
    )

    plt.xlabel(
        "Per-protein Pearson correlation"
    )

    plt.ylabel(
        "Number of proteins"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ANALYSIS_DIR,
            "per_protein_Pearson_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# GNN INFERENCE RUNTIME
# ============================================================

def benchmark_inference_runtime(
    dataset,
    best_params
):

    if len(dataset) == 0:

        return None

    model = NodeMLP_GCN(

        dataset[0].x.size(1),

        hidden_dim=best_params[
            "hidden_dim"
        ],

        num_gcn_layers=best_params[
            "num_gcn_layers"
        ],

        dropout=best_params[
            "dropout"
        ],

        use_residual=best_params[
            "residual"
        ],

        use_batch_norm=best_params[
            "batch_norm"
        ],

        activation=best_params[
            "activation"
        ],

        use_bias=best_params[
            "use_bias"
        ]

    ).to(device)

    model.eval()

    records = []

    print("\n")
    print("==============================================")
    print("GNN INFERENCE RUNTIME")
    print("==============================================")

    with torch.no_grad():

        for data in dataset:

            batch = Batch.from_data_list(
                [data]
            ).to(device)

            # ------------------------------------------------
            # Warmup
            # ------------------------------------------------

            _ = model(
                batch.x,
                batch.edge_index
            )

            if device == "cuda":

                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(
                batch.x,
                batch.edge_index
            )

            if device == "cuda":

                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                -
                start
            )

            records.append({

                "PDB_Name":
                    data.pdb_name,

                "N_residues":
                    data.num_nodes,

                "Inference_seconds":
                    elapsed
            })

    df = pd.DataFrame(
        records
    )

    output = os.path.join(
        ANALYSIS_DIR,
        "inference_runtime_per_structure.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    mean_time = (
        df[
            "Inference_seconds"
        ].mean()
    )

    median_time = (
        df[
            "Inference_seconds"
        ].median()
    )

    print(
        f"Structures: "
        f"{len(df)}"
    )

    print(
        f"Mean inference: "
        f"{mean_time:.6f} s"
    )

    print(
        f"Median inference: "
        f"{median_time:.6f} s"
    )

    if mean_time > 0:

        print(
            f"Structures/hour: "
            f"{3600 / mean_time:.0f}"
        )

    print(
        f"\nSaved:\n{output}"
    )

    return df


# ============================================================
# PROTEOME-SCALE ESTIMATE
# ============================================================

def estimate_proteome_scale_runtime():

    preprocessing_file = os.path.join(
        ANALYSIS_DIR,
        "preprocessing_runtime_per_structure.csv"
    )

    inference_file = os.path.join(
        ANALYSIS_DIR,
        "inference_runtime_per_structure.csv"
    )

    if not os.path.exists(
        preprocessing_file
    ):

        print(
            "Preprocessing runtime file missing."
        )

        return

    if not os.path.exists(
        inference_file
    ):

        print(
            "Inference runtime file missing."
        )

        return

    prep = pd.read_csv(
        preprocessing_file
    )

    inference = pd.read_csv(
        inference_file
    )

    prep_mean = (
        prep[
            "Preprocessing_seconds"
        ].mean()
    )

    inference_mean = (
        inference[
            "Inference_seconds"
        ].mean()
    )

    total_per_structure = (
        prep_mean
        +
        inference_mean
    )

    print("\n")
    print("==============================================")
    print("PROTEOME-SCALE FEASIBILITY")
    print("==============================================")

    print(
        f"Mean preprocessing: "
        f"{prep_mean:.4f} s/structure"
    )

    print(
        f"Mean inference: "
        f"{inference_mean:.6f} s/structure"
    )

    print(
        f"Total: "
        f"{total_per_structure:.4f} "
        f"s/structure"
    )

    rows = []

    for n in [
        1000,
        5000,
        10000,
        20000,
        50000,
        100000
    ]:

        prep_hours = (
            prep_mean
            *
            n
            /
            3600
        )

        inference_hours = (
            inference_mean
            *
            n
            /
            3600
        )

        total_hours = (
            total_per_structure
            *
            n
            /
            3600
        )

        rows.append({

            "N_structures":
                n,

            "Preprocessing_hours":
                prep_hours,

            "Inference_hours":
                inference_hours,

            "Total_hours":
                total_hours,

            "Total_days":
                total_hours / 24
        })

        print(
            f"{n:,} proteins:"
        )

        print(
            f"  preprocessing = "
            f"{prep_hours:.2f} h"
        )

        print(
            f"  inference     = "
            f"{inference_hours:.2f} h"
        )

        print(
            f"  total         = "
            f"{total_hours:.2f} h "
            f"({total_hours / 24:.2f} days)"
        )

    output = os.path.join(
        ANALYSIS_DIR,
        "proteome_scale_runtime.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved:\n{output}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # YOUR ORIGINAL SETUP
    # ========================================================

    folder = os.getcwd()

    print(
        f"\nWorking directory:\n"
        f"{folder}"
    )

    # ========================================================
    # BUILD DATASET
    # ========================================================

    dataset = (
        build_dataset_from_folder(
            folder
        )
    )

    print(
        f"\nBuilt "
        f"{len(dataset)} "
        f"protein graphs."
    )

    # ========================================================
    # GRAPH ANALYSIS
    # ========================================================

    analyze_graph_connectivity(
        dataset
    )

    # ========================================================
    # FLEXIBILITY CLASS COUNTS
    # ========================================================

    flexible_count, non_flexible_count = (
        count_flexible_residues(
            dataset,
            threshold=0.0
        )
    )

    print(
        f"\nTotal flexible residues: "
        f"{flexible_count}"
    )

    print(
        f"Total non-flexible residues: "
        f"{non_flexible_count}"
    )

    # ========================================================
    # OPTUNA
    # ========================================================

    print("\n")
    print("==============================================")
    print("HYPERPARAMETER OPTIMIZATION")
    print("==============================================")

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        lambda trial:
        train_and_evaluate(
            trial,
            dataset
        ),
        n_trials=25
    )

    trial = study.best_trial

    print(
        "\nBest hyperparameters:"
    )

    for key, value in (
        trial.params.items()
    ):

        print(
            f"  {key}: {value}"
        )

    # ========================================================
    # SAVE HYPERPARAMETERS
    # ========================================================

    with open(
        "best_hyperparameters.txt",
        "w"
    ) as f:

        f.write(
            "Best trial hyperparameters:\n"
        )

        for key, value in (
            trial.params.items()
        ):

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            f"\nBest trial value: "
            f"{trial.value:.6f}\n"
        )

    # ========================================================
    # 10-FOLD CROSS VALIDATION
    # ========================================================

    cv_results = (
        run_cross_validation(
            dataset,
            trial.params,
            n_splits=10
        )
    )

    # ========================================================
    # REVIEWER ANALYSIS
    # ========================================================

    print("\n\n")
    print("########################################################")
    print("#")
    print("# REVIEWER 2 ANALYSIS")
    print("#")
    print("########################################################")

    # ========================================================
    # 1. PER-FOLD
    #
    # RMSE
    # MAE
    # R2
    # Pearson
    # Spearman
    # ========================================================

    calculate_fold_metrics_from_csv()

    # ========================================================
    # 2. PER-CLASS
    #
    # Flexible vs Non-flexible
    # ========================================================

    calculate_per_class_metrics()

    # ========================================================
    # 3. PER-PROTEIN
    #
    # This is particularly important for Reviewer 2.
    # ========================================================

    calculate_per_protein_metrics()

    # ========================================================
    # 4. FAILURE CASES
    # ========================================================

    identify_failure_cases(
        top_n=20
    )

    # ========================================================
    # 5. ERROR DISTRIBUTION PLOTS
    # ========================================================

    plot_error_distributions()

    # ========================================================
    # 6. GNN INFERENCE RUNTIME
    # ========================================================

    benchmark_inference_runtime(
        dataset,
        trial.params
    )

    # ========================================================
    # 7. PROTEOME-SCALE FEASIBILITY
    # ========================================================

    estimate_proteome_scale_runtime()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("########################################################")
    print("# ANALYSIS COMPLETE")
    print("########################################################")

    print(
        "\nReviewer analysis files are in:"
    )

    print(
        os.path.abspath(
            ANALYSIS_DIR
        )
    )

    print("\nGenerated files include:")

    for filename in sorted(
        os.listdir(
            ANALYSIS_DIR
        )
    ):

        print(
            f"  {filename}"
        )
