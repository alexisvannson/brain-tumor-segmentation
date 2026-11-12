"""Inspect raw H5 masks to decide `data.mask_format`.

* One-hot labels (NCR, ED, ET): channels never overlap -> mask_format: labels
* Nested regions (WT ⊇ TC ⊇ ET): channels overlap      -> mask_format: regions

Usage: python scripts/check_masks.py <data_dir> [--n 2000]
"""
import argparse
import os
import random
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.splits import list_h5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--n", type=int, default=2000, help="Number of files to sample")
    args = parser.parse_args()

    files = list_h5(args.data_dir)
    files = random.Random(0).sample(files, min(args.n, len(files)))

    voxels = np.zeros(3, dtype=np.int64)
    overlap = 0         # voxels with more than one channel set
    nested = 0          # voxels where c2 ⊂ c1 ⊂ c0 holds strictly with overlap
    fg = 0
    values = set()
    for path in files:
        with h5py.File(path, "r") as f:
            m = f["mask"][:]
        m = m.reshape(-1, m.shape[-1]) if m.shape[-1] == 3 else m.reshape(3, -1).T
        values.update(np.unique(m).tolist())
        b = m > 0
        voxels += b.sum(axis=0)
        n_set = b.sum(axis=1)
        fg += int((n_set > 0).sum())
        overlap += int((n_set > 1).sum())
        nested += int((b[:, 2] & b[:, 1] & b[:, 0]).sum())

    print(f"Files sampled:          {len(files)}")
    print(f"Unique mask values:     {sorted(values)[:10]}")
    print(f"Foreground voxels:      {fg}")
    print(f"Voxels per channel:     {voxels.tolist()}")
    print(f"Voxels in >1 channel:   {overlap} ({overlap / max(fg, 1):.2%} of foreground)")
    if fg == 0:
        print("No foreground found; sample more files.")
    elif overlap == 0:
        print("=> Channels are disjoint: one-hot labels. Use  mask_format: labels")
    else:
        print(f"=> Channels overlap ({nested} voxels set in all three): nested regions. "
              "Use  mask_format: regions")


if __name__ == "__main__":
    main()
