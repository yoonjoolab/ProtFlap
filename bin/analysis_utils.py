import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from scipy.stats import pearsonr, spearmanr


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_OUTPUT_DIR = "reviewer_analysis"

DEFAULT_PROTEOME_SIZES = [
    1000,
    5000,
    10000,
    50000,
    100000,
]


# ============================================================
# SAFE CORRELATIONS
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

    if np.std(y_true) == 0:
        return np.nan

    if np.std(y_pred) == 0:
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

    if np.std(y_true) == 0:
        return np.nan

    if np.std(y_pred) == 0:
        return np.nan

    return spearmanr(
        y_true,
        y_pred
    )[0]


# ============================================================
# REGRESSION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

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

    n = len(y_true)

    if n == 0:

        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Pearson": np.nan,
            "Spearman": np.nan
        }

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

    if n >= 2:

        try:
            r2 = r2_score(
                y_true,
                y_pred
            )
        except Exception:
            r2 = np.nan

    else:

        r2 = np.nan

    return {

        "N": n,

        "RMSE": rmse,

        "MAE": mae,

        "R2": r2,

        "Pearson": safe_pearson(
            y_true,
            y_pred
        ),

        "Spearman": safe_spearman(
            y_true,
            y_pred
        )
    }


# ============================================================
# BUILD DATAFRAME
# ============================================================

def build_prediction_dataframe(
    y_true,
    y_pred,
    pdb_names,
    residue_indices,
    interfaces=None
):

    df = pd.DataFrame({

        "PDB_Name":
            np.asarray(
                pdb_names
            ),

        "Residue_Index":
            np.asarray(
                residue_indices
            ),

        "Target_RMSF":
            np.asarray(
                y_true,
                dtype=float
            ),

        "Predicted_RMSF":
            np.asarray(
                y_pred,
                dtype=float
            )
    })

    # --------------------------------------------------------
    # Interface
    # --------------------------------------------------------

    if interfaces is None:

        df["Interface"] = np.nan

    else:

        df["Interface"] = pd.to_numeric(
            np.asarray(
                interfaces
            ),
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    df = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Target_RMSF",
            "Predicted_RMSF"
        ]
    ).copy()

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    df["Error"] = (
        df["Predicted_RMSF"]
        -
        df["Target_RMSF"]
    )

    df["Absolute_Error"] = np.abs(
        df["Error"]
    )

    df["Squared_Error"] = (
        df["Error"] ** 2
    )

    # --------------------------------------------------------
    # Binary flexibility
    #
    # RMSF_norm >= 0 = Flexible
    # RMSF_norm < 0  = Non-flexible
    # --------------------------------------------------------

    df["Target_Binary"] = (
        df["Target_RMSF"] >= 0
    ).astype(int)

    df["Predicted_Binary"] = (
        df["Predicted_RMSF"] >= 0
    ).astype(int)

    df["Target_Class"] = np.where(
        df["Target_RMSF"] >= 0,
        "Flexible",
        "Non-flexible"
    )

    df["Predicted_Class"] = np.where(
        df["Predicted_RMSF"] >= 0,
        "Flexible",
        "Non-flexible"
    )

    return df


# ============================================================
# OVERALL ANALYSIS
# ============================================================

def analyze_overall(
    df,
    output_dir=DEFAULT_OUTPUT_DIR
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    metrics = calculate_metrics(
        df["Target_RMSF"].values,
        df["Predicted_RMSF"].values
    )

    metrics["Proteins"] = (
        df["PDB_Name"].nunique()
    )

    metrics["Residues"] = len(df)

    # Put important columns first
    result = pd.DataFrame([{

        "Proteins":
            metrics["Proteins"],

        "Residues":
            metrics["Residues"],

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
    }])

    result.to_csv(
        os.path.join(
            output_dir,
            "overall_metrics.csv"
        ),
        index=False
    )

    with open(
        os.path.join(
            output_dir,
            "overall_metrics.txt"
        ),
        "w"
    ) as f:

        f.write(
            "==============================================\n"
        )

        f.write(
            "ProtFlap OVERALL TEST RESULTS\n"
        )

        f.write(
            "==============================================\n\n"
        )

        f.write(
            f"Proteins: {metrics['Proteins']}\n"
        )

        f.write(
            f"Residues: {metrics['Residues']}\n\n"
        )

        f.write(
            f"RMSE:     {metrics['RMSE']:.6f}\n"
        )

        f.write(
            f"MAE:      {metrics['MAE']:.6f}\n"
        )

        f.write(
            f"R2:       {metrics['R2']:.6f}\n"
        )

        f.write(
            f"Pearson:  {metrics['Pearson']:.6f}\n"
        )

        f.write(
            f"Spearman: {metrics['Spearman']:.6f}\n"
        )

    return result


# ============================================================
# PER-RESIDUE OUTPUT
# ============================================================

def save_per_residue_results(
    df,
    output_dir=DEFAULT_OUTPUT_DIR
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Complete predictions
    # --------------------------------------------------------

    prediction_file = os.path.join(
        output_dir,
        "per_residue_predictions.csv"
    )

    df.to_csv(
        prediction_file,
        index=False
    )

    # --------------------------------------------------------
    # Error CSV
    # --------------------------------------------------------

    error_columns = [

        "PDB_Name",
        "Residue_Index",

        "Target_RMSF",
        "Predicted_RMSF",

        "Error",
        "Absolute_Error",
        "Squared_Error",

        "Target_Class",
        "Predicted_Class",

        "Target_Binary",
        "Predicted_Binary"
    ]

    error_df = df[
        error_columns
    ].copy()

    error_file = os.path.join(
        output_dir,
        "per_residue_errors.csv"
    )

    error_df.to_csv(
        error_file,
        index=False
    )

    print(
        f"Saved: {prediction_file}"
    )

    print(
        f"Saved: {error_file}"
    )

    return error_df


# ============================================================
# PER-PROTEIN METRICS
# ============================================================

def analyze_per_protein(
    df,
    output_dir=DEFAULT_OUTPUT_DIR
):

    rows = []

    for pdb_name, group in df.groupby(
        "PDB_Name"
    ):

        y_true = (
            group["Target_RMSF"]
            .values
        )

        y_pred = (
            group["Predicted_RMSF"]
            .values
        )

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        # ----------------------------------------------------
        # Error statistics
        # ----------------------------------------------------

        errors = (
            y_pred -
            y_true
        )

        abs_errors = np.abs(
            errors
        )

        # ----------------------------------------------------
        # Flexible / non-flexible
        # ----------------------------------------------------

        flexible = (
            group["Target_RMSF"]
            >= 0
        )

        n_flexible = flexible.sum()

        n_nonflexible = (
            (~flexible).sum()
        )

        # ----------------------------------------------------
        # Binary classification
        # ----------------------------------------------------

        y_true_binary = (
            group["Target_Binary"]
            .values
        )

        y_pred_binary = (
            group["Predicted_Binary"]
            .values
        )

        accuracy = accuracy_score(
            y_true_binary,
            y_pred_binary
        )

        precision = precision_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0
        )

        recall = recall_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0
        )

        f1 = f1_score(
            y_true_binary,
            y_pred_binary,
            zero_division=0
        )

        rows.append({

            "PDB_Name":
                pdb_name,

            "N_residues":
                metrics["N"],

            "N_flexible":
                n_flexible,

            "N_nonflexible":
                n_nonflexible,

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

            "Median_Absolute_Error":
                np.median(abs_errors),

            "Max_Absolute_Error":
                np.max(abs_errors),

            "RMSE":
                metrics["RMSE"],

            "MAE":
                metrics["MAE"],

            "R2":
                metrics["R2"],

            "Pearson":
                metrics["Pearson"],

            "Spearman":
                metrics["Spearman"],

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1
        })

    protein_df = pd.DataFrame(
        rows
    )

    protein_file = os.path.join(
        output_dir,
        "per_protein_metrics.csv"
    )

    protein_df.to_csv(
        protein_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary across proteins
    # --------------------------------------------------------

    metric_columns = [

        "RMSE",
        "MAE",
        "R2",
        "Pearson",
        "Spearman",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    summary_rows = []

    for metric in metric_columns:

        summary_rows.append({

            "Metric":
                metric,

            "Mean":
                protein_df[
                    metric
                ].mean(),

            "Median":
                protein_df[
                    metric
                ].median(),

            "SD":
                protein_df[
                    metric
                ].std(),

            "Min":
                protein_df[
                    metric
                ].min(),

            "Max":
                protein_df[
                    metric
                ].max()
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = os.path.join(
        output_dir,
        "per_protein_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print(
        f"Saved: {protein_file}"
    )

    print(
        f"Saved: {summary_file}"
    )

    return (
        protein_df,
        summary_df
    )


# ============================================================
# FLEXIBLE / NON-FLEXIBLE
# ============================================================

def analyze_classes(
    df,
    output_dir=DEFAULT_OUTPUT_DIR
):

    rows = []

    for class_name in [
        "Flexible",
        "Non-flexible"
    ]:

        if class_name == "Flexible":

            mask = (
                df["Target_RMSF"]
                >= 0
            )

        else:

            mask = (
                df["Target_RMSF"]
                < 0
            )

        sub = df[
            mask
        ]

        if len(sub) == 0:
            continue

        metrics = calculate_metrics(
            sub["Target_RMSF"].values,
            sub["Predicted_RMSF"].values
        )

        rows.append({

            "Class":
                class_name,

            "N":
                len(sub),

            "RMSE":
                metrics["RMSE"],

            "MAE":
                metrics["MAE"],

            "R2":
                metrics["R2"],

            "Pearson":
                metrics["Pearson"],

            "Spearman":
                metrics["Spearman"],

            "Mean_Error":
                sub["Error"].mean(),

            "Mean_Absolute_Error":
                sub["Absolute_Error"].mean()
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            output_dir,
            "class_metrics.csv"
        ),
        index=False
    )

    return result


# ============================================================
# BINARY CLASSIFICATION
# ============================================================

def analyze_binary_classification(
    df,
    output_dir=DEFAULT_OUTPUT_DIR
):

    y_true = (
        df["Target_Binary"]
        .values
    )

    y_pred = (
        df["Predicted_Binary"]
        .values
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    result = pd.DataFrame([{

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp,

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
    }])

    result.to_csv(
        os.path.join(
            output_dir,
            "binary_classification.csv"
        ),
        index=False
    )

    return result


# ============================================================
# INTERFACE ANALYSIS
# ============================================================

def analyze_interface(
    df,
    output_dir=DEFAULT_OUTPUT_DIR
):

    # --------------------------------------------------------
    # Check whether interface information exists
    # --------------------------------------------------------

    if (
        "Interface" not in df.columns
        or
        df["Interface"].notna().sum() == 0
    ):

        print()
        print(
            "⚠️ Interface information was "
            "not found in the dataset."
        )

        print(
            "Interface analysis skipped."
        )

        result = pd.DataFrame([{

            "Region":
                "Interface information unavailable",

            "N":
                0,

            "RMSE":
                np.nan,

            "MAE":
                np.nan,

            "R2":
                np.nan,

            "Pearson":
                np.nan,

            "Spearman":
                np.nan
        }])

        result.to_csv(
            os.path.join(
                output_dir,
                "interface_metrics.csv"
            ),
            index=False
        )

        return result

    # --------------------------------------------------------
    # Convert interface values
    # --------------------------------------------------------

    interface_values = (
        pd.to_numeric(
            df["Interface"],
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # Assume 1 = interface
    #        0 = non-interface
    # --------------------------------------------------------

    masks = {

        "Interface":
            interface_values == 1,

        "Non-interface":
            interface_values == 0
    }

    rows = []

    for region, mask in masks.items():

        sub = df[
            mask
        ]

        if len(sub) < 1:
            continue

        metrics = calculate_metrics(
            sub["Target_RMSF"].values,
            sub["Predicted_RMSF"].values
        )

        rows.append({

            "Region":
                region,

            "N":
                len(sub),

            "RMSE":
                metrics["RMSE"],

            "MAE":
                metrics["MAE"],

            "R2":
                metrics["R2"],

            "Pearson":
                metrics["Pearson"],

            "Spearman":
                metrics["Spearman"],

            "Mean_Error":
                sub["Error"].mean(),

            "Mean_Absolute_Error":
                sub["Absolute_Error"].mean()
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            output_dir,
            "interface_metrics.csv"
        ),
        index=False
    )

    return result


# ============================================================
# RUNTIME ANALYSIS
# ============================================================

def analyze_runtime(
    n_proteins,
    preprocessing_time,
    inference_time,
    output_dir=DEFAULT_OUTPUT_DIR
):

    total_time = (
        preprocessing_time
        +
        inference_time
    )

    mean_preprocessing = (
        preprocessing_time
        /
        n_proteins
    )

    mean_inference = (
        inference_time
        /
        n_proteins
    )

    mean_total = (
        total_time
        /
        n_proteins
    )

    result = pd.DataFrame([{

        "Proteins":
            n_proteins,

        "Preprocessing_seconds":
            preprocessing_time,

        "Inference_seconds":
            inference_time,

        "Total_seconds":
            total_time,

        "Mean_preprocessing_seconds_per_protein":
            mean_preprocessing,

        "Mean_inference_seconds_per_protein":
            mean_inference,

        "Mean_total_seconds_per_protein":
            mean_total,

        "Preprocessing_hours":
            preprocessing_time / 3600,

        "Inference_hours":
            inference_time / 3600,

        "Total_hours":
            total_time / 3600,

        "Total_days":
            total_time / 86400
    }])

    result.to_csv(
        os.path.join(
            output_dir,
            "runtime_summary.csv"
        ),
        index=False
    )

    return result


# ============================================================
# PROTEOME-SCALE ANALYSIS
# ============================================================

def analyze_proteome_scale(
    mean_preprocessing_seconds,
    mean_inference_seconds,
    output_dir=DEFAULT_OUTPUT_DIR,
    protein_sizes=None
):

    if protein_sizes is None:

        protein_sizes = (
            DEFAULT_PROTEOME_SIZES
        )

    rows = []

    for n in protein_sizes:

        preprocessing = (
            n
            *
            mean_preprocessing_seconds
        )

        inference = (
            n
            *
            mean_inference_seconds
        )

        total = (
            preprocessing
            +
            inference
        )

        rows.append({

            "Proteins":
                n,

            "Preprocessing_seconds":
                preprocessing,

            "Inference_seconds":
                inference,

            "Total_seconds":
                total,

            "Preprocessing_hours":
                preprocessing / 3600,

            "Inference_hours":
                inference / 3600,

            "Total_hours":
                total / 3600,

            "Total_days":
                total / 86400
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            output_dir,
            "proteome_scale_runtime.csv"
        ),
        index=False
    )

    return result


# ============================================================
# PER-PROTEIN RUNTIME
# ============================================================

def analyze_per_protein_runtime(
    protein_names,
    preprocessing_total,
    inference_total,
    output_dir=DEFAULT_OUTPUT_DIR
):

    protein_names = np.asarray(
        protein_names
    )

    unique_proteins = np.unique(
        protein_names
    )

    n = len(
        unique_proteins
    )

    if n == 0:

        return pd.DataFrame()

    mean_preprocessing = (
        preprocessing_total / n
    )

    mean_inference = (
        inference_total / n
    )

    rows = []

    for protein in unique_proteins:

        rows.append({

            "PDB_Name":
                protein,

            "Estimated_preprocessing_seconds":
                mean_preprocessing,

            "Estimated_inference_seconds":
                mean_inference,

            "Estimated_total_seconds":
                (
                    mean_preprocessing
                    +
                    mean_inference
                )
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        os.path.join(
            output_dir,
            "per_protein_runtime.csv"
        ),
        index=False
    )

    return result


# ============================================================
# PLOTS
# ============================================================

def make_plots(
    df,
    output_dir=DEFAULT_OUTPUT_DIR
):

    y_true = (
        df["Target_RMSF"]
        .values
    )

    y_pred = (
        df["Predicted_RMSF"]
        .values
    )

    # --------------------------------------------------------
    # Scatter
    # --------------------------------------------------------

    pearson = safe_pearson(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.35,
        s=10
    )

    minimum = min(
        y_true.min(),
        y_pred.min()
    )

    maximum = max(
        y_true.max(),
        y_pred.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        "Target RMSF_norm"
    )

    plt.ylabel(
        "Predicted RMSF_norm"
    )

    plt.title(
        f"ProtFlap Test Set "
        f"(Pearson r = {pearson:.3f})"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "test_scatter.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Error distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        df["Error"].values,
        bins=50,
        edgecolor="black"
    )

    plt.xlabel(
        "Prediction Error"
    )

    plt.ylabel(
        "Number of residues"
    )

    plt.title(
        "ProtFlap Prediction Error"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "error_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Target vs prediction distribution
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.hist(
        y_true,
        bins=50,
        alpha=0.5,
        label="Target"
    )

    plt.hist(
        y_pred,
        bins=50,
        alpha=0.5,
        label="Predicted"
    )

    plt.xlabel(
        "RMSF_norm"
    )

    plt.ylabel(
        "Number of residues"
    )

    plt.title(
        "Target vs Predicted RMSF"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "target_prediction_distribution.png"
        ),
        dpi=300
    )

    plt.close()

def run_complete_analysis(
    y_true,
    y_pred,
    pdb_names,
    residue_indices,
    interfaces,
    preprocessing_time,
    inference_time,
    output_dir=DEFAULT_OUTPUT_DIR,
    proteome_sizes=None
):

    print()
    print("=" * 70)
    print("RUNNING COMPLETE ANALYSIS")
    print("=" * 70)


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # BUILD DATAFRAME
    # ========================================================

    df = build_prediction_dataframe(
        y_true,
        y_pred,
        pdb_names,
        residue_indices,
        interfaces
    )

    # ========================================================
    # NUMBER OF PROTEINS
    # ========================================================

    n_proteins = df["PDB_Name"].nunique()

    if n_proteins == 0:

        raise ValueError(
            "No proteins found in prediction dataframe."
        )

    # ========================================================
    # OVERALL ANALYSIS
    # ========================================================

    overall = analyze_overall(
        df,
        output_dir
    )

    # ========================================================
    # PER-RESIDUE RESULTS
    # ========================================================

    save_per_residue_results(
        df,
        output_dir
    )

    # ========================================================
    # PER-PROTEIN ANALYSIS
    # ========================================================

    (
        protein_df,
        protein_summary
    ) = analyze_per_protein(
        df,
        output_dir
    )

    # ========================================================
    # FLEXIBILITY CLASS ANALYSIS
    # ========================================================

    class_df = analyze_classes(
        df,
        output_dir
    )

    # ========================================================
    # BINARY CLASSIFICATION
    # ========================================================

    binary_df = analyze_binary_classification(
        df,
        output_dir
    )

    # ========================================================
    # INTERFACE ANALYSIS
    # ========================================================

    interface_df = analyze_interface(
        df,
        output_dir
    )

    # ========================================================
    # RUNTIME ANALYSIS
    # ========================================================

    runtime_df = analyze_runtime(

        n_proteins=n_proteins,

        preprocessing_time=preprocessing_time,

        inference_time=inference_time,

        output_dir=output_dir
    )

    # ========================================================
    # CALCULATE MEAN TIMES
    # ========================================================

    mean_preprocessing = (
        preprocessing_time
        /
        n_proteins
    )

    mean_inference = (
        inference_time
        /
        n_proteins
    )

    mean_total = (
        mean_preprocessing
        +
        mean_inference
    )

    # ========================================================
    # PROTEOME-SCALE ANALYSIS
    # ========================================================

    proteome_df = analyze_proteome_scale(

        mean_preprocessing,

        mean_inference,

        output_dir,

        protein_sizes=proteome_sizes
    )

    # ========================================================
    # PER-PROTEIN RUNTIME
    # ========================================================

    per_protein_runtime = analyze_per_protein_runtime(

        df["PDB_Name"].values,

        preprocessing_time,

        inference_time,

        output_dir
    )

    # ========================================================
    # GENERATE PLOTS
    # ========================================================

    make_plots(
        df,
        output_dir
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    print(
        f"Proteins: {n_proteins}"
    )

    print(
        f"Residues: {len(df)}"
    )

    print()

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    print(
        f"RMSE: "
        f"{overall.iloc[0]['RMSE']:.4f}"
    )

    print(
        f"MAE: "
        f"{overall.iloc[0]['MAE']:.4f}"
    )

    print(
        f"R²: "
        f"{overall.iloc[0]['R2']:.4f}"
    )

    print(
        f"Pearson: "
        f"{overall.iloc[0]['Pearson']:.4f}"
    )

    print(
        f"Spearman: "
        f"{overall.iloc[0]['Spearman']:.4f}"
    )

    # --------------------------------------------------------
    # RUNTIME
    # --------------------------------------------------------

    print()

    print(
        f"Mean preprocessing: "
        f"{mean_preprocessing:.6f} s/protein"
    )

    print(
        f"Mean inference: "
        f"{mean_inference:.6f} s/protein"
    )

    print(
        f"Mean total: "
        f"{mean_total:.6f} s/protein"
    )

    # --------------------------------------------------------
    # PROTEOME-SCALE RESULTS
    # --------------------------------------------------------

    print()

    print(
        "Proteome-scale:"
    )

    print()

    print(
        proteome_df.to_string(
            index=False
        )
    )

    # ========================================================
    # OUTPUT LOCATION
    # ========================================================

    print()
    print("=" * 70)

    print(
        "Analysis files saved in:"
    )

    print(
        os.path.abspath(
            output_dir
        )
    )

    print("=" * 70)

    # ========================================================
    # RETURN ALL RESULTS
    # ========================================================

    return {

        "dataframe":
            df,

        "overall":
            overall,

        "per_protein":
            protein_df,

        "per_protein_summary":
            protein_summary,

        "class_metrics":
            class_df,

        "binary":
            binary_df,

        "interface":
            interface_df,

        "runtime":
            runtime_df,

        "proteome":
            proteome_df,

        "per_protein_runtime":
            per_protein_runtime
    }
