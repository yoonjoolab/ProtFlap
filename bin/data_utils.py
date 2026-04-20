import logging
import pandas as pd

def load_rmsf_data(pdb_name):
    """Load rmsf_norm values from CSV, indexed by residue index."""
    csv_file = f"{pdb_name}.csv"
    try:
        df = pd.read_csv(csv_file)
        if "rmsf_norm" not in df.columns:
            logging.error(f"'rmsf_norm' column not found in {csv_file}")
            return None
        return df["rmsf_norm"].values
    except FileNotFoundError:
        logging.error(f"CSV file {csv_file} not found.")
        return None

