"""Format-level tests for src/data_loader.py using SYNTHETIC files.

These verify the parser/validator logic only. They say nothing about the real
UCI data; run `python -m src.data_loader` for that.
Run:  python tests/test_data_loader.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_loader as dl  # noqa: E402


def write_fake_batches(root: Path, counts: dict[int, int], seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for b, n in counts.items():
        with open(root / f"batch{b}.dat", "w") as fh:
            for _ in range(n):
                cls = int(rng.integers(1, 7))
                conc = float(rng.choice([10, 50, 100, 500, 1000]))
                vals = rng.normal(size=dl.N_FEATURES)
                toks = " ".join(f"{i+1}:{v:.6f}" for i, v in enumerate(vals))
                fh.write(f"{cls};{conc:.6f} {toks}\n")


def test_valid_structure():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "nested").mkdir()  # files may live in a subfolder
        write_fake_batches(root / "nested", dl.EXPECTED_BATCH_COUNTS)
        df = dl.load_dataframe(root, validate=True)
        assert df.shape == (13910, 132), df.shape
        assert df.columns[:4].tolist() == ["Batch", "Class", "Gas", "Concentration"]
        assert df.columns[4] == "Sensor1_DR" and df.columns[-1] == "Sensor16_EMAd0.1"
        assert df.iloc[:, 4:].isna().sum().sum() == 0
    print("PASS valid structure -> 13910 x 132")


def test_wrong_counts_stop():
    bad = dict(dl.EXPECTED_BATCH_COUNTS)
    bad[3] = 1500  # corrupt one batch
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        write_fake_batches(root, bad)
        try:
            dl.load_dataframe(root, validate=True)
        except dl.DatasetValidationError as exc:
            assert "batch3" in str(exc) and "13910" in str(exc)
            print("PASS count mismatch raises DatasetValidationError")
            return
        raise AssertionError("validator did not stop on bad counts")


def test_bad_line_and_missing_index():
    try:
        dl.parse_line("1 1:2.0", 1, "x")
        raise AssertionError("missing ';' accepted")
    except dl.ParseError:
        pass
    _, _, v = dl.parse_line("2;50.0 1:1.5 128:3.5", 1, "x")
    assert v[0] == 1.5 and v[127] == 3.5 and np.isnan(v[1])
    print("PASS malformed line rejected; missing indices -> NaN (not 0)")


def test_feature_names():
    names = dl.build_feature_names()
    assert len(names) == 128 and len(set(names)) == 128
    assert names[:8] == ["Sensor1_DR", "Sensor1_abs_DR", "Sensor1_EMAi0.001",
                         "Sensor1_EMAi0.01", "Sensor1_EMAi0.1",
                         "Sensor1_EMAd0.001", "Sensor1_EMAd0.01", "Sensor1_EMAd0.1"]
    print("PASS 128 unique feature names")


if __name__ == "__main__":
    test_feature_names()
    test_bad_line_and_missing_index()
    test_valid_structure()
    test_wrong_counts_stop()
    print("\nAll parser tests passed.")
