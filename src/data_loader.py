"""
data_loader.py
==============
Download, parse, name and VALIDATE the UCI "Gas Sensor Array Drift Dataset
at Different Concentrations" (UCI id 270).

Source page:
    https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

Raw format (libsvm-like, one MEASUREMENT per line):
    <class>;<concentration> 1:<v1> 2:<v2> ... 128:<v128>

    class         : 1..6  (1 Ethanol, 2 Ethylene, 3 Ammonia,
                           4 Acetaldehyde, 5 Acetone, 6 Toluene)
    concentration : gas concentration in ppmv
    128 features  : 16 sensors x 8 derived features (sensor-major order)

IMPORTANT: gas class is NOT an anomaly label. Anomalies are injected later.

Usage:
    python -m src.data_loader                 # download + parse + validate + save
    python -m src.data_loader --no-download   # use files already in data/raw
    python -m src.data_loader --diagnose      # print raw-file diagnostics only
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARSED_CSV = PROCESSED_DIR / "gas_sensor_drift_parsed.csv"

UCI_PAGE = ("https://archive.ics.uci.edu/dataset/270/"
            "gas+sensor+array+drift+dataset+at+different+concentrations")
# Static download endpoint used by the UCI repository for dataset id 270.
UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/270/"
               "gas+sensor+array+drift+dataset+at+different+concentrations.zip")

N_SENSORS = 16
N_DERIVED = 8
N_FEATURES = N_SENSORS * N_DERIVED  # 128

GAS_NAMES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
             4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}

EXPECTED_BATCH_COUNTS = {1: 445, 2: 1244, 3: 1586, 4: 161, 5: 197,
                         6: 2300, 7: 3613, 8: 294, 9: 470, 10: 3600}
EXPECTED_TOTAL = 13_910
EXPECTED_COLUMNS = 4 + N_FEATURES  # Batch, Class, Gas, Concentration + 128

# Order of the 8 derived features per sensor, as documented by UCI / Vergara
# et al. (2012): steady-state change dR, its absolute value, and EMA transient
# features (increasing 'i' and decreasing 'd') at three smoothing constants.
# NOTE: verify against the "Variables" table on the UCI page (see README).
DERIVED_FEATURE_NAMES = [
    "DR", "abs_DR",
    "EMAi0.001", "EMAi0.01", "EMAi0.1",
    "EMAd0.001", "EMAd0.01", "EMAd0.1",
]


class DatasetValidationError(RuntimeError):
    """Raised when parsed data does not match the documented dataset."""


class ParseError(ValueError):
    """Raised when a raw line cannot be parsed."""


# --------------------------------------------------------------------------- #
# Feature naming
# --------------------------------------------------------------------------- #
def build_feature_names() -> list[str]:
    """['Sensor1_DR', 'Sensor1_abs_DR', ..., 'Sensor16_EMAd0.1'] (128 names)."""
    return [f"Sensor{s}_{d}"
            for s in range(1, N_SENSORS + 1)
            for d in DERIVED_FEATURE_NAMES]


def feature_mapping_table() -> pd.DataFrame:
    """Documentation table: original Feature index -> sensor, derived feature."""
    rows = []
    for i, name in enumerate(build_feature_names(), start=1):
        sensor, derived = name.split("_", 1)
        rows.append({"original_index": i, "feature_name": name,
                     "sensor": int(sensor.replace("Sensor", "")),
                     "derived_feature": derived})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _extract_zip_recursive(zip_path: Path, dest: Path) -> None:
    """Extract a zip; if it contains further zips, extract those too."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    for nested in list(dest.rglob("*.zip")):
        if nested.resolve() == zip_path.resolve():
            continue
        target = nested.parent / nested.stem
        target.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(target)
        except zipfile.BadZipFile:
            print(f"[warn] could not extract nested archive {nested}")


def find_batch_files(raw_dir: Path = RAW_DIR) -> dict[int, Path]:
    """Locate batch1.dat ... batch10.dat anywhere under raw_dir."""
    found: dict[int, Path] = {}
    for p in raw_dir.rglob("*"):
        m = re.fullmatch(r"batch(\d+)\.dat", p.name, flags=re.IGNORECASE)
        if m and p.is_file():
            found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def download_dataset(raw_dir: Path = RAW_DIR, url: str = UCI_ZIP_URL,
                     force: bool = False) -> dict[int, Path]:
    """Download the official UCI zip, extract it, return {batch_no: path}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = find_batch_files(raw_dir)
    if len(existing) == 10 and not force:
        print(f"[download] batch files already present in {raw_dir}; skipping.")
        return existing

    zip_path = raw_dir / "gas_sensor_array_drift_dataset.zip"
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, \
                open(zip_path, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception as exc:  # network / HTTP errors
        raise RuntimeError(
            f"Automatic download failed: {exc}\n"
            f"Fallback: download the ZIP manually from\n  {UCI_PAGE}\n"
            f"and extract it into {raw_dir} (batch1.dat ... batch10.dat)."
        ) from exc

    print(f"[download] saved {zip_path.name} ({zip_path.stat().st_size/1e6:.1f} MB)")
    _extract_zip_recursive(zip_path, raw_dir)
    files = find_batch_files(raw_dir)
    if len(files) != 10:
        raise DatasetValidationError(
            f"Expected 10 batch files after extraction, found {sorted(files)}. "
            f"Run `python -m src.data_loader --diagnose` and inspect {raw_dir}.")
    return files


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_line(line: str, lineno: int = 0, source: str = "") -> tuple[int, float, np.ndarray]:
    """Parse one 'class;conc idx:val idx:val ...' line -> (class, conc, 128-vector).

    Missing feature indices become NaN (never silently zero)."""
    where = f"{source}:{lineno}"
    head, _, rest = line.strip().partition(" ")
    if ";" not in head:
        raise ParseError(f"{where}: header token '{head[:30]}' lacks 'class;concentration'")
    cls_s, conc_s = head.split(";", 1)
    try:
        cls, conc = int(float(cls_s)), float(conc_s)
    except ValueError as exc:
        raise ParseError(f"{where}: bad class/concentration '{head}'") from exc

    values = np.full(N_FEATURES, np.nan, dtype=np.float64)
    for tok in rest.split():
        try:
            idx_s, val_s = tok.split(":", 1)
            idx, val = int(idx_s), float(val_s)
        except ValueError as exc:
            raise ParseError(f"{where}: bad feature token '{tok}'") from exc
        if not 1 <= idx <= N_FEATURES:
            raise ParseError(f"{where}: feature index {idx} outside 1..{N_FEATURES}")
        values[idx - 1] = val
    return cls, conc, values


def parse_batch_file(path: Path, batch_no: int) -> pd.DataFrame:
    """Parse a single batch file into a DataFrame (one row per measurement)."""
    classes, concs, mats = [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue  # skip blank lines, but they are reported by diagnose
            c, k, v = parse_line(line, lineno, path.name)
            classes.append(c)
            concs.append(k)
            mats.append(v)
    df = pd.DataFrame(np.vstack(mats) if mats else np.empty((0, N_FEATURES)),
                      columns=build_feature_names())
    df.insert(0, "Concentration", concs)
    df.insert(0, "Gas", [GAS_NAMES.get(c, f"Unknown({c})") for c in classes])
    df.insert(0, "Class", classes)
    df.insert(0, "Batch", batch_no)
    return df


def load_dataframe(raw_dir: Path = RAW_DIR, validate: bool = True) -> pd.DataFrame:
    """Parse all batches (in batch order, original row order preserved)."""
    files = find_batch_files(raw_dir)
    if len(files) != 10:
        raise DatasetValidationError(
            f"Found batch files {sorted(files)} in {raw_dir}; expected 1..10.")
    df = pd.concat([parse_batch_file(p, b) for b, p in files.items()],
                   ignore_index=True)
    df["Batch"] = df["Batch"].astype("int64")
    df["Class"] = df["Class"].astype("int64")
    if validate:
        validate_dataframe(df, raw_dir)
    return df


# --------------------------------------------------------------------------- #
# Diagnostics + validation
# --------------------------------------------------------------------------- #
def diagnose_files(raw_dir: Path = RAW_DIR, n_preview: int = 2) -> None:
    """Print raw-file facts used to decide whether the parser's assumptions hold."""
    files = find_batch_files(raw_dir)
    print(f"Batch files found: {sorted(files)}")
    for b, p in files.items():
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        non_empty = [l for l in lines if l.strip()]
        tok_counts = Counter(len(l.split()) for l in non_empty)
        has_semicolon = sum(";" in l.split(" ", 1)[0] for l in non_empty)
        print(f"\n[batch{b}] {p.name}: physical lines={len(lines)}, "
              f"non-empty={len(non_empty)}, expected={EXPECTED_BATCH_COUNTS.get(b)}")
        print(f"  tokens-per-line distribution (expect 129 = header + 128): "
              f"{dict(tok_counts.most_common(5))}")
        print(f"  lines with 'class;conc' header: {has_semicolon}/{len(non_empty)}")
        for l in non_empty[:n_preview]:
            print("  preview:", l[:110] + (" ..." if len(l) > 110 else ""))


def validate_dataframe(df: pd.DataFrame, raw_dir: Path | None = None) -> dict:
    """Hard validation. Raises DatasetValidationError on any structural mismatch."""
    problems: list[str] = []
    report: dict = {}

    # 1. total rows
    report["total_rows"] = len(df)
    if len(df) != EXPECTED_TOTAL:
        problems.append(f"total rows {len(df)} != {EXPECTED_TOTAL}")

    # 2. per-batch counts
    got = df["Batch"].value_counts().sort_index().to_dict()
    report["batch_counts"] = got
    for b, exp in EXPECTED_BATCH_COUNTS.items():
        if got.get(b) != exp:
            problems.append(f"batch{b}: got {got.get(b)}, expected {exp}")

    # 3. schema
    if df.shape[1] != EXPECTED_COLUMNS:
        problems.append(f"{df.shape[1]} columns != {EXPECTED_COLUMNS}")
    expected_cols = ["Batch", "Class", "Gas", "Concentration"] + build_feature_names()
    if list(df.columns) != expected_cols:
        problems.append("column names/order differ from expected schema")

    # 4. value sanity
    if not set(df["Class"].unique()).issubset(GAS_NAMES):
        problems.append(f"unexpected class labels: {sorted(df['Class'].unique())}")
    if not (df["Concentration"] > 0).all():
        problems.append("non-positive concentration values present")
    feats = df[build_feature_names()]
    report["n_nan"] = int(feats.isna().sum().sum())
    report["n_inf"] = int(np.isinf(feats.to_numpy()).sum())
    if report["n_inf"]:
        problems.append(f"{report['n_inf']} infinite feature values")
    # NaNs are reported (warning), not fatal: the original data is expected complete.
    report["duplicate_rows"] = int(df.duplicated().sum())

    if problems:
        msg = ["DATASET VALIDATION FAILED - do NOT continue to anomaly injection."]
        msg += [f"  - {p}" for p in problems]
        msg.append("Run `python -m src.data_loader --diagnose` and inspect the raw "
                   "files; adapt the parser to the real format instead of forcing counts.")
        raise DatasetValidationError("\n".join(msg))

    report["status"] = "OK"
    return report


def print_validation_summary(df: pd.DataFrame, report: dict) -> None:
    print("\n" + "=" * 62)
    print("VALIDATION PASSED")
    print("=" * 62)
    print(f"Shape                : {df.shape}  (expected 13910 x 132)")
    print(f"NaN feature values   : {report['n_nan']}")
    print(f"Duplicate rows       : {report['duplicate_rows']}")
    print("Rows per batch       :")
    for b, n in report["batch_counts"].items():
        print(f"   batch{b:<2d} {n:>5d}  (expected {EXPECTED_BATCH_COUNTS[b]})")
    print("Rows per gas         :")
    print(df["Gas"].value_counts().to_string())
    print("\nSchema (first 8 + last 2 columns):")
    print(df.dtypes.iloc[list(range(8)) + [-2, -1]].to_string())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args(argv)

    if args.diagnose:
        diagnose_files(args.raw_dir)
        return 0
    if not args.no_download:
        download_dataset(args.raw_dir, force=args.force_download)
    try:
        df = load_dataframe(args.raw_dir, validate=False)
        report = validate_dataframe(df, args.raw_dir)
    except (DatasetValidationError, ParseError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print("---- raw-file diagnostics ----", file=sys.stderr)
        diagnose_files(args.raw_dir)
        return 1

    print_validation_summary(df, report)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PARSED_CSV, index=False)
    feature_mapping_table().to_csv(PROCESSED_DIR / "feature_mapping.csv", index=False)
    print(f"\nSaved: {PARSED_CSV}\nSaved: {PROCESSED_DIR / 'feature_mapping.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
