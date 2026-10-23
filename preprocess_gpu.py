
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
import pty
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TINKER PATHS AND SETTINGS
# ============================================================
# Replace this with the absolute path to the Tinker-GPU directory.
TINKER_DIR = Path("/path/to/tinker-gpu").resolve()

TINKER_SOURCE = TINKER_DIR / "tinker" / "source"
TINKER_BUILD = TINKER_DIR / "build"

PDBXYZ = (TINKER_SOURCE / "pdbxyz.x").resolve()
XYZPDB = (TINKER_SOURCE / "xyzpdb.x").resolve()
TINKER9 = (TINKER_BUILD / "tinker9").resolve()
ANALYZE = (TINKER_SOURCE / "analyze.x").resolve()
FORCE_KEY = (TINKER_BUILD / "force.key").resolve()

MIN_GRID_DEFAULT = "0.1"

ENERGY_COLUMNS = [
    "EB", "EA", "EBA", "EUB", "EAA", "EOPB", "EOPD", "EID",
    "EIT", "ET", "EPT", "EBT", "EAT", "ETT", "EV", "ER",
    "EDSP", "EC", "ECD", "ED", "EM", "EP", "ECT", "ERXF",
    "ES", "ELF", "EG", "EX",
]


# ============================================================
# ENVIRONMENT / LOGGING
# ============================================================
def configure_cuda_library_path() -> None:
    """Add CUDA libraries bundled with the active Conda/Python environment."""
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")

    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib")
        candidates.append(
            Path(conda_prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "nvidia"
        )

    site_roots = [
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
        Path(sys.prefix)
        / "local"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
    ]

    for root in site_roots:
        nvidia = root / "nvidia"
        if nvidia.is_dir():
            for component in nvidia.iterdir():
                lib_dir = component / "lib"
                if lib_dir.is_dir():
                    candidates.append(lib_dir)

    if conda_prefix:
        nvidia = (
            Path(conda_prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "nvidia"
        )
        if nvidia.is_dir():
            for component in nvidia.iterdir():
                lib_dir = component / "lib"
                if lib_dir.is_dir():
                    candidates.append(lib_dir)

    paths: list[str] = []

    for path in candidates:
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in paths:
                paths.append(resolved)

    for path in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if path and path not in paths:
            paths.append(path)

    os.environ["LD_LIBRARY_PATH"] = ":".join(paths)


def elapsed_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours} h {minutes} min {secs} sec"
    if minutes:
        return f"{minutes} min {secs} sec"
    return f"{secs} sec"


def check_installation() -> None:
    for label, path in [
        ("pdbxyz.x", PDBXYZ),
        ("xyzpdb.x", XYZPDB),
        ("tinker9", TINKER9),
        ("analyze.x", ANALYZE),
        ("force.key", FORCE_KEY),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Tinker file does not exist: {path}"
            )

        if label != "force.key" and not os.access(path, os.X_OK):
            raise PermissionError(
                f"Tinker executable is not executable: {path}"
            )


def run_command(
    command: list[str],
    cwd: Path,
    *,
    input_text: str | None = None,
    stdout_path: Path | None = None,
    description: str = "command",
) -> None:
    """Run a command and include captured output if it fails."""
    output_handle = (
        open(stdout_path, "w", encoding="utf-8", errors="replace")
        if stdout_path
        else None
    )

    try:
        result = subprocess.run(
            [str(x) for x in command],
            cwd=str(cwd),
            input=input_text,
            text=True,
            stdout=output_handle if output_handle else subprocess.PIPE,
            stderr=subprocess.STDOUT if output_handle else subprocess.PIPE,
            check=False,
            env=os.environ.copy(),
        )
    finally:
        if output_handle:
            output_handle.close()

    if result.returncode != 0:
        if not stdout_path:
            captured = (result.stdout or "") + "\n" + (result.stderr or "")
        elif stdout_path.exists():
            captured = stdout_path.read_text(
                encoding="utf-8", errors="replace"
            )
        else:
            captured = ""

        raise RuntimeError(
            f"{description} failed (exit code {result.returncode}).\n"
            f"{captured[-3000:]}"
        )


def run_xyzpdb(
    command: list[str],
    cwd: Path,
    description: str,
) -> None:
    """Run xyzpdb.x using a pseudo-terminal and select PDB format."""
    master_fd, slave_fd = pty.openpty()
    process = None
    output_tail = ""
    answered_format = False

    try:
        process = subprocess.Popen(
            [str(x) for x in command],
            cwd=str(cwd),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=os.environ.copy(),
            close_fds=True,
        )

        os.close(slave_fd)
        slave_fd = -1

        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.25)

            if ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    chunk = b""

                if chunk:
                    decoded = chunk.decode("utf-8", errors="replace")
                    output_tail = (output_tail + decoded)[-4000:]

                    if (
                        not answered_format
                        and "PDB or CIF Format" in output_tail
                    ):
                        os.write(master_fd, b"PDB\r\n")
                        answered_format = True

                elif process.poll() is not None:
                    break

            if process.poll() is not None:
                try:
                    while True:
                        ready, _, _ = select.select(
                            [master_fd], [], [], 0
                        )
                        if not ready:
                            break

                        chunk = os.read(master_fd, 4096)
                        if not chunk:
                            break

                        decoded = chunk.decode(
                            "utf-8", errors="replace"
                        )
                        output_tail = (output_tail + decoded)[-4000:]

                except OSError:
                    pass

                break

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, command
            )

        if (
            "PDB or CIF Format" in output_tail
            and not answered_format
        ):
            raise RuntimeError(
                "xyzpdb.x showed the PDB/CIF prompt, "
                "but it was not answered."
            )

    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()

        if slave_fd >= 0:
            os.close(slave_fd)

        os.close(master_fd)


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not created: {path}"
        )
    return path


def safe_unlink(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
    except OSError:
        pass


# ============================================================
# CONVERSION / MINIMIZATION
# ============================================================
def convert_pdb_to_xyz(
    pdb_file: Path,
    work_dir: Path,
) -> Path:
    run_command(
        [PDBXYZ, pdb_file.name, "-k", FORCE_KEY],
        work_dir,
        input_text="PDB\n",
        description=f"Converting {pdb_file.name} to XYZ",
    )

    return require_file(
        work_dir / f"{pdb_file.stem}.xyz",
        "PDB-to-XYZ conversion",
    )


def convert_xyz_to_pdb(
    xyz_file: Path,
    work_dir: Path,
) -> Path:
    run_xyzpdb(
        [XYZPDB, xyz_file.name, "-k", FORCE_KEY],
        work_dir,
        "Converting XYZ to PDB",
    )

    candidates = [
        work_dir / f"{xyz_file.stem}.pdb_2",
        work_dir / f"{xyz_file.stem}.pdb",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"xyzpdb.x finished but no PDB output was found. "
        f"Checked: {candidates}"
    )


def add_hydrogens(
    input_pdb: Path,
    work_dir: Path,
    base: str,
) -> Path:
    xyz_file = convert_pdb_to_xyz(input_pdb, work_dir)
    generated_pdb = convert_xyz_to_pdb(xyz_file, work_dir)

    hydro_pdb = work_dir / f"{base}_hydro.pdb"

    if generated_pdb.resolve() != hydro_pdb.resolve():
        if hydro_pdb.exists():
            hydro_pdb.unlink()
        generated_pdb.rename(hydro_pdb)

    return require_file(hydro_pdb, "Hydrogen-added PDB")


def minimize_structure(
    hydro_pdb: Path,
    work_dir: Path,
    base: str,
    min_grid: str,
) -> Path:
    hydro_xyz = convert_pdb_to_xyz(hydro_pdb, work_dir)

    run_command(
        [
            TINKER9,
            "minimize",
            hydro_xyz.name,
            "-k",
            FORCE_KEY,
            min_grid,
        ],
        work_dir,
        description="Tinker9 minimization",
    )

    minimized_xyz = require_file(
        work_dir / f"{hydro_xyz.name}_2",
        "Tinker9 minimized XYZ",
    )

    generated_min_pdb = convert_xyz_to_pdb(
        minimized_xyz, work_dir
    )

    min_pdb = work_dir / f"{base}_min.pdb"

    if generated_min_pdb.resolve() != min_pdb.resolve():
        if min_pdb.exists():
            min_pdb.unlink()
        generated_min_pdb.rename(min_pdb)

    require_file(min_pdb, "Minimized PDB")

    if min_pdb.stat().st_size < 1000:
        raise RuntimeError(
            f"Minimized PDB looks unexpectedly small "
            f"({min_pdb.stat().st_size} bytes): {min_pdb}"
        )

    return min_pdb


# ============================================================
# ENERGY EXTRACTION
# ============================================================
def parse_energy_breakdown(
    txt_file: Path,
) -> list[list[str]]:
    """Parse Tinker Potential Energy Breakdown over Atoms output."""
    lines = txt_file.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    start = None

    for i, line in enumerate(lines):
        if "Potential Energy Breakdown over Atoms" in line:
            start = i + 1
            break

    if start is None:
        raise RuntimeError(
            f"Could not find 'Potential Energy Breakdown over Atoms' "
            f"in {txt_file}. Check the Tinker analyze output."
        )

    cleaned = [
        line.strip()
        for line in lines[start:]
        if line.strip()
    ]

    records: list[list[str]] = []
    current: list[str] = []

    for line in cleaned:
        values = line.split()

        is_atom_start = (
            len(values) >= 7
            and values[0].lstrip("+-").isdigit()
        )

        if is_atom_start and current:
            records.append(current)
            current = []

        current.extend(values)

        if len(current) >= 29:
            records.append(current[:29])
            current = current[29:]

    if current:
        records.append(current)

    records = [
        row for row in records
        if len(row) == 29
        and row[0].lstrip("+-").isdigit()
    ]

    if not records:
        raise RuntimeError(
            f"No complete atom energy rows (29 values each) "
            f"parsed from {txt_file}. "
            "The output layout may differ from the expected Tinker format."
        )

    return records


def residue_map_from_pdb(
    pdb_file: Path,
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    with pdb_file.open(
        "r", encoding="utf-8", errors="replace"
    ) as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                atom_serial = line[6:11].strip()
                residue_number = line[22:26].strip()

                if atom_serial:
                    mapping[atom_serial] = residue_number

    return mapping


def extract_energies(
    min_pdb: Path,
    work_dir: Path,
) -> tuple[Path, Path]:
    base = min_pdb.stem

    output_csv = work_dir / f"{base}_energy.csv"
    averaged_csv = work_dir / f"{base}.csv"
    txt_file = work_dir / f"{base}.txt"

    xyz_file = convert_pdb_to_xyz(min_pdb, work_dir)

    run_command(
        [ANALYZE, xyz_file.name, "-k", FORCE_KEY],
        work_dir,
        input_text="A\nALL\n",
        stdout_path=txt_file,
        description="Tinker analyze.x",
    )

    require_file(txt_file, "Tinker analyze output")

    if txt_file.stat().st_size == 0:
        raise RuntimeError(
            f"Tinker analyze output is empty: {txt_file}"
        )

    rows = parse_energy_breakdown(txt_file)
    residue_map = residue_map_from_pdb(min_pdb)

    header = ["Residue", "Atom"] + ENERGY_COLUMNS

    with output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for row in rows:
            atom_index = str(row[0])
            residue = residue_map.get(atom_index, "")
            numeric_terms = row[1:]

            if len(numeric_terms) != len(ENERGY_COLUMNS):
                continue

            writer.writerow(
                [residue, atom_index] + numeric_terms
            )

    df = pd.read_csv(output_csv)

    if df.empty:
        raise RuntimeError(
            f"Atom-level energy CSV has no data rows: {output_csv}"
        )

    for col in ENERGY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    averaged_df = (
        df.dropna(subset=["Residue"])
        .groupby("Residue", sort=False)[ENERGY_COLUMNS]
        .mean()
        .reset_index()
    )

    if averaged_df.empty:
        raise RuntimeError(
            f"No per-residue averages could be calculated from "
            f"{output_csv}. Check that PDB atom serials correspond "
            "to Tinker atom indices."
        )

    averaged_df.to_csv(
        averaged_csv,
        index=False,
        float_format="%.4f",
    )

    return output_csv, averaged_csv


# ============================================================
# PROCESS EACH PDB
# ============================================================
def process_one(
    pdb_path: Path,
    output_dir: Path | None,
    min_grid: str,
) -> dict:
    file_start = time.perf_counter()
    base = pdb_path.stem
    work_dir = pdb_path.parent

    minimization_seconds = np.nan
    energy_seconds = np.nan
    status = "FAILED"
    final_files: list[Path] = []

    print(f"\nProcessing {pdb_path.name}...", flush=True)

    try:
        minimize_start = time.perf_counter()

        hydro_pdb = add_hydrogens(
            pdb_path, work_dir, base
        )

        min_pdb = minimize_structure(
            hydro_pdb, work_dir, base, min_grid
        )

        minimization_seconds = (
            time.perf_counter() - minimize_start
        )

        final_files.append(min_pdb)

        print(
            f"Minimization finished in "
            f"{elapsed_text(minimization_seconds)} "
            f"({minimization_seconds:.2f} seconds)",
            flush=True,
        )

        energy_start = time.perf_counter()

        atom_csv, averaged_csv = extract_energies(
            min_pdb, work_dir
        )

        energy_seconds = time.perf_counter() - energy_start

        final_files.extend([atom_csv, averaged_csv])

        print(
            f"Energy extraction finished in "
            f"{elapsed_text(energy_seconds)} "
            f"({energy_seconds:.2f} seconds)",
            flush=True,
        )

        intermediates = [
            work_dir / f"{base}.xyz",
            work_dir / f"{base}.seq",
            hydro_pdb,
            work_dir / f"{base}_hydro.xyz",
            work_dir / f"{base}_hydro.seq",
            work_dir / f"{base}_hydro.xyz_2",
            work_dir / f"{base}_min.xyz",
            work_dir / f"{base}_min.seq",
            work_dir / f"{base}_min.txt",
            work_dir / f"{base}.txt",
        ]

        for path in intermediates:
            safe_unlink(path)

        if output_dir:
            output_dir.mkdir(
                parents=True, exist_ok=True
            )

            moved_files: list[Path] = []

            for path in final_files:
                if not path.exists():
                    continue

                destination = output_dir / path.name

                if path.resolve() != destination.resolve():
                    if destination.exists():
                        destination.unlink()

                    shutil.move(
                        str(path), str(destination)
                    )

                moved_files.append(destination)

            final_files = moved_files

        status = "SUCCESS"

    except Exception as exc:
        status = f"FAILED: {type(exc).__name__}: {exc}"
        print(
            f"\nERROR processing {pdb_path.name}: {status}",
            file=sys.stderr,
        )

    total_seconds = time.perf_counter() - file_start

    print(
        f"Finished {pdb_path.name} | "
        f"Total time: {elapsed_text(total_seconds)} "
        f"({total_seconds:.2f} seconds) | Status: {status}",
        flush=True,
    )

    return {
        "PDB_Name": pdb_path.name,
        "Minimization_Time_sec": minimization_seconds,
        "Energy_Extraction_Time_sec": energy_seconds,
        "Total_Processing_Time_sec": total_seconds,
        "Total_Processing_Time_readable": elapsed_text(
            total_seconds
        ),
        "Status": status,
    }


# ============================================================
# MAIN
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "ProtFlap Tinker-GPU preprocessing, minimization, "
            "and energy extraction"
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help=(
            "Input PDB file(s) or wildcard(s), "
            "e.g. -i /path/to/input.pdb or "
            "-i '/path/to/*.pdb'"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Optional output directory for minimized PDB "
            "and energy CSV files"
        ),
    )

    parser.add_argument(
        "--min-grid",
        default=MIN_GRID_DEFAULT,
        help=(
            "Tinker9 minimization convergence criterion "
            f"(default: {MIN_GRID_DEFAULT})"
        ),
    )

    args = parser.parse_args()

    configure_cuda_library_path()
    check_installation()

    pdb_paths: list[Path] = []

    for item in args.input:
        matches = glob.glob(item)

        if not matches and Path(item).is_file():
            matches = [item]

        pdb_paths.extend(
            Path(match).expanduser().resolve()
            for match in matches
        )

    pdb_paths = sorted({
        p for p in pdb_paths
        if p.is_file() and p.suffix.lower() == ".pdb"
    })

    if not pdb_paths:
        print(
            f"ERROR: No PDB files found for input: {args.input}",
            file=sys.stderr,
        )
        return 2

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else None
    )

    if output_dir:
        output_dir.mkdir(
            parents=True, exist_ok=True
        )

    print(
        f"Processing {len(pdb_paths)} PDB file(s).",
        flush=True,
    )

    overall_start = time.perf_counter()
    results: list[dict] = []

    for pdb_path in pdb_paths:
        results.append(
            process_one(
                pdb_path,
                output_dir,
                str(args.min_grid),
            )
        )

    overall_seconds = time.perf_counter() - overall_start

    timing_dir = output_dir if output_dir else Path.cwd()
    timing_dir.mkdir(parents=True, exist_ok=True)

    timing_csv = timing_dir / "processing_times.csv"

    pd.DataFrame(results).to_csv(
        timing_csv,
        index=False,
        float_format="%.4f",
    )

    successes = sum(
        row["Status"] == "SUCCESS"
        for row in results
    )
    failures = len(results) - successes

    print("\nFINAL SUMMARY")
    print(
        f"Files: {len(results)} | "
        f"Successful: {successes} | Failed: {failures}"
    )
    print(
        f"Total elapsed time: {elapsed_text(overall_seconds)} "
        f"({overall_seconds:.2f} seconds)"
    )
    print(f"Timing report: {timing_csv}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

