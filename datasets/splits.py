"""Patient-level train / val / test splits.

The Kaggle BraTS 2020 export stores one axial slice per file
(``volume_{patient}_slice_{index}.h5``). Splitting at the file level puts
neighbouring slices of the same patient in both train and validation, which
leaks information. Everything here works on *patient IDs* instead.
"""
import json
import os
import random
import re
from collections import defaultdict

SLICE_PATTERN = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")


def list_h5(data_dir):
    return sorted(
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")
    )


def parse_slice_name(path):
    """Return (patient_id, slice_index) for slice files, or (stem, None) otherwise.

    Files that don't follow the slice naming scheme (e.g. one-volume-per-file
    datasets) are treated as their own patient.
    """
    name = os.path.basename(path)
    m = SLICE_PATTERN.search(name)
    if m:
        return m.group(1), int(m.group(2))
    return os.path.splitext(name)[0], None


def group_by_patient(files):
    """Map patient_id -> list of files, slices ordered by slice index."""
    groups = defaultdict(list)
    for f in files:
        pid, idx = parse_slice_name(f)
        groups[pid].append((idx if idx is not None else 0, f))
    return {pid: [f for _, f in sorted(items)] for pid, items in sorted(groups.items())}


def make_patient_split(patient_ids, val_frac=0.15, test_frac=0.15, seed=42):
    """Shuffle patient IDs with a fixed seed and cut them into three disjoint sets."""
    ids = sorted(patient_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    return {
        "test":  sorted(ids[:n_test]),
        "val":   sorted(ids[n_test:n_test + n_val]),
        "train": sorted(ids[n_test + n_val:]),
    }


def load_or_create_split(data_dir, split_file, val_frac=0.15, test_frac=0.15, seed=42):
    """Load a saved split, or create and save one.

    The split is written to disk once and then reused, so every run (training,
    tuning, final evaluation) sees exactly the same held-out patients.
    """
    if split_file and os.path.exists(split_file):
        with open(split_file) as f:
            return json.load(f)

    patients = group_by_patient(list_h5(data_dir))
    split = make_patient_split(patients.keys(), val_frac, test_frac, seed)
    split["meta"] = {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_patients": len(patients),
    }
    if split_file:
        os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Created patient split ({len(split['train'])}/{len(split['val'])}/"
              f"{len(split['test'])} train/val/test) -> {split_file}")
    return split
