#!/usr/bin/env python3
import subprocess
from pathlib import Path
import glob
import os
import csv
import pandas as pd
import argparse
import shutil

# ------------------------------
# Tinker paths & parameters
# ------------------------------
tinker_dir = Path("/path/to/Tinker")
force = "path/to/force.key"
min_grid = "0.01"

pdb2xyz = tinker_dir / "pdbxyz"
xyz2pdb = tinker_dir / "xyzpdb"
minimize = tinker_dir / "minimize"
analyze = tinker_dir / "analyze"

# Energy columns for per-residue averaging
energy_columns = [
    'EB', 'EA', 'EBA', 'EUB', 'EAA', 'EOPB', 'EOPD', 'EID',
    'EIT', 'ET', 'EPT', 'EBT', 'EAT', 'ETT', 'EV', 'ER',
    'EDSP', 'EC', 'ECD', 'ED', 'EM', 'EP', 'ECT', 'ERXF',
    'ES', 'ELF', 'EG', 'EX'
]

# ------------------------------
# Argument parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Tinker PDB Minimization + Energy Analysis")
parser.add_argument("-i", "--input", nargs="+", required=True, help="Input PDB file(s) or wildcards")
parser.add_argument("-o", "--output", default=None, help="Optional output directory for final min PDBs and CSVs")
args = parser.parse_args()

# Expand input wildcards
pdb_files = []
for item in args.input:
    pdb_files += glob.glob(item)
if not pdb_files:
    print("❌ No PDB files found for input:", args.input)
    exit(1)

# Output folder (optional)
outdir = Path(args.output) if args.output else None
if outdir:
    outdir.mkdir(exist_ok=True, parents=True)

# ------------------------------
# Main loop: Minimize & Analyze
# ------------------------------
for pdb_path in pdb_files:
    pdb_path = Path(pdb_path)
    if not pdb_path.exists() or pdb_path.suffix.lower() != ".pdb":
        print(f"⚠️ Skipping invalid file: {pdb_path}")
        continue

    base = pdb_path.stem
    work_dir = pdb_path.parent  # always work in the same folder as input PDB

    print("="*60)
    print(f"🔹 Minimizing: {pdb_path}")
    print(f"   Working dir: {work_dir}")
    print("="*60)

    # ------------------------------
    # Step 1: Add hydrogens
    # ------------------------------
    subprocess.run([str(pdb2xyz), str(pdb_path), "-k", str(force), "ALL", "A", "ALL"], check=True)
    subprocess.run([str(xyz2pdb), f"{base}.xyz", "-k", str(force)], check=True)

    hydro_pdb = work_dir / f"{base}_hydro.pdb"
    if Path(f"{base}.pdb_2").exists():
        Path(f"{base}.pdb_2").rename(hydro_pdb)
    else:
        print(f"❌ Hydrogen-added PDB not found for {base}")
        continue

    subprocess.run([str(pdb2xyz), str(hydro_pdb), "-k", str(force), "ALL", "A", "ALL"], check=True)

    # ------------------------------
    # Step 2: Minimize
    # ------------------------------
    hydro_xyz = work_dir / f"{base}_hydro.xyz"
    subprocess.run([str(minimize), str(hydro_xyz), "-k", str(force), min_grid], check=True)
    subprocess.run([str(xyz2pdb), f"{hydro_xyz}_2", "-k", str(force)], check=True)

    # Rename minimized PDB
    min_pdb = work_dir / f"{base}_min.pdb"
    if Path(f"{base}_hydro.pdb_2").exists():
        Path(f"{base}_hydro.pdb_2").rename(min_pdb)
    else:
        print(f"❌ Minimized PDB not found for {base}")
        continue

    # Cleanup intermediate files
    for f in [
        Path(f"{base}.xyz"),
        Path(f"{base}.seq"),
        hydro_pdb,
        Path(f"{base}_hydro.seq"),
        hydro_xyz,
        Path(f"{base}_hydro.xyz_2"),
    ]:
        if f.exists():
            f.unlink()

    print(f"✅ Minimized PDB created: {min_pdb}")

    # ------------------------------
    # Step 3: Energy Analysis
    # ------------------------------
    if min_pdb.stat().st_size < 1000:
        print(f"⚠️ Skipping small PDB file: {min_pdb}")
        continue

    pdb_base = min_pdb.stem
    output_csv = work_dir / f"{pdb_base}_energy.csv"  # Atom-level CSV
    averaged_csv = work_dir / f"{pdb_base}.csv"       # Per-residue CSV

    # Convert minimized PDB -> XYZ
    subprocess.run([pdb2xyz, str(min_pdb), "-k", str(force)], check=True)
    xyz_file = work_dir / f"{pdb_base}.xyz"
    txt_file = work_dir / f"{pdb_base}.txt"

    # Run Tinker analyze
    with open(txt_file, "w") as f:
        subprocess.run([analyze, str(xyz_file), "-k", str(force), "A"], stdout=f, check=True)

    if not txt_file.exists() or txt_file.stat().st_size == 0:
        print(f"❌ analyze output missing for {min_pdb}")
        continue

    # Clean analyze TXT output
    with open(txt_file, "r") as f:
        lines = f.readlines()

    cleaned_lines = []
    skip = True
    for line in lines:
        if "Potential Energy Breakdown over Atoms :" in line:
            skip = False
            continue
        if not skip and line.strip():
            cleaned_lines.append(line.strip())

    # TXT -> Atom-level CSV
    residues = {}
    with open(min_pdb, "r") as pdb_f:
        for line in pdb_f:
            if line.startswith(("ATOM", "HETATM")):
                atom_num = line[6:11].strip()
                res_num = line[22:26].strip()
                residues[atom_num] = res_num

    atom_data = []
    with open(output_csv, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        for line in cleaned_lines:
            values = line.split()
            if len(values) >= 7:
                if atom_data:
                    atom_num = atom_data[0]
                    writer.writerow([residues.get(atom_num, "Residue")] + atom_data)
                atom_data = [values[0]] + values[1:]
            else:
                atom_data.extend(values)
            if len(atom_data) == 29:
                atom_num = atom_data[0]
                writer.writerow([residues.get(atom_num, "Residue")] + atom_data)
                atom_data = []
        if atom_data:
            atom_num = atom_data[0]
            writer.writerow([residues.get(atom_num, "Residue")] + atom_data)

    # Average per-residue
    df = pd.read_csv(output_csv)
    df['Residue'] = df['Residue'].ffill()
    available_columns = [c for c in energy_columns if c in df.columns]

    if not available_columns:
        print(f"⚠️ No matching energy columns found for {min_pdb}, skipping averaging.")
        continue

    averaged_df = df.groupby('Residue')[available_columns].mean().reset_index()
    averaged_df.to_csv(averaged_csv, index=False, float_format='%.4f')

    # Cleanup intermediate files
    for f in [xyz_file, txt_file, work_dir / f"{pdb_base}.seq"]:
        if f.exists():
            f.unlink()

    # ------------------------------
    # Step 4: Move final files if -o specified
    # ------------------------------
    if outdir:
        for f in [min_pdb, output_csv, averaged_csv]:
            dest = outdir / f.name
            shutil.move(str(f), dest)

    print(f"✅ Finished processing {pdb_path}\n")

print("🎉 All PDBs minimized and analyzed successfully.")

