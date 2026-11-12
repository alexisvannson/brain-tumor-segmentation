"""Tests for patient-level splits, label conversion, normalisation and metrics."""
import json

import numpy as np
import pytest
import torch

from datasets.splits import group_by_patient, load_or_create_split, make_patient_split, parse_slice_name
from datasets.transforms import labels_to_regions, prepare_mask, zscore
from scripts.metrics import HD95_EMPTY_PENALTY, case_metrics, dice_iou_per_case, hd95, summarize


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

class TestSplits:
    def test_parse_slice_name(self):
        assert parse_slice_name("/d/volume_41_slice_100.h5") == ("41", 100)
        assert parse_slice_name("/d/patient_7.h5") == ("patient_7", None)

    def test_group_orders_slices_numerically(self):
        files = [f"volume_1_slice_{i}.h5" for i in (10, 2, 1)] + ["volume_2_slice_0.h5"]
        groups = group_by_patient(files)
        assert groups["1"] == ["volume_1_slice_1.h5", "volume_1_slice_2.h5", "volume_1_slice_10.h5"]
        assert list(groups) == ["1", "2"]

    def test_split_disjoint_and_complete(self):
        ids = [str(i) for i in range(100)]
        s = make_patient_split(ids, val_frac=0.15, test_frac=0.15, seed=0)
        train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
        assert not (train & val) and not (train & test) and not (val & test)
        assert train | val | test == set(ids)
        assert len(test) == 15 and len(val) == 15

    def test_split_deterministic_and_seed_dependent(self):
        ids = [str(i) for i in range(50)]
        assert make_patient_split(ids, seed=1) == make_patient_split(list(reversed(ids)), seed=1)
        assert make_patient_split(ids, seed=1)["test"] != make_patient_split(ids, seed=2)["test"]

    def test_split_file_created_then_reused(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        for p in range(10):
            (data / f"volume_{p}_slice_0.h5").write_bytes(b"")
        path = tmp_path / "split.json"
        s1 = load_or_create_split(str(data), str(path), seed=3)
        assert path.exists() and s1["meta"]["n_patients"] == 10
        # A different seed must not silently change an existing split
        s2 = load_or_create_split(str(data), str(path), seed=99)
        assert s1 == s2 == json.loads(path.read_text())


# ---------------------------------------------------------------------------
# label conversion / normalisation
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_labels_to_regions(self):
        # one voxel each: NCR, ED, ET, background
        m = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
        r = labels_to_regions(m)
        assert r[0].tolist() == [1, 1, 1, 0]  # WT
        assert r[1].tolist() == [1, 0, 1, 0]  # TC
        assert r[2].tolist() == [0, 0, 1, 0]  # ET

    def test_regions_are_nested(self):
        m = (torch.rand(3, 8, 8, 8) > 0.7).float()
        r = labels_to_regions(m).bool()
        assert (r[2] <= r[1]).all() and (r[1] <= r[0]).all()

    def test_prepare_mask_passthrough_and_error(self):
        m = torch.rand(3, 4, 4)
        assert prepare_mask(m, "regions") is m
        with pytest.raises(ValueError):
            prepare_mask(m, "nope")

    def test_zscore_ignores_background(self):
        img = torch.zeros(1, 10, 10)
        img[0, 2:8, 2:8] = torch.arange(36, dtype=torch.float32).reshape(6, 6) + 100
        out = zscore(img, spatial_dims=(1, 2))
        brain = img != 0
        assert (out[~brain] == 0).all()
        assert abs(out[brain].mean().item()) < 1e-5
        assert abs(out[brain].std(unbiased=False).item() - 1) < 1e-4

    def test_zscore_empty_slice_stays_zero(self):
        out = zscore(torch.zeros(4, 5, 5), spatial_dims=(1, 2))
        assert (out == 0).all() and not out.isnan().any()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _b(x):
    return torch.tensor(x, dtype=torch.bool)


class TestMetrics:
    def test_dice_perfect_and_disjoint(self):
        t = _b([[[[1, 1, 0, 0]]]])
        d, i = dice_iou_per_case(t, t)
        assert d.item() == 1 and i.item() == 1
        d, i = dice_iou_per_case(_b([[[[0, 0, 1, 1]]]]), t)
        assert d.item() == 0 and i.item() == 0

    def test_dice_partial(self):
        d, i = dice_iou_per_case(_b([[[[1, 1, 1, 0]]]]), _b([[[[1, 0, 0, 0]]]]))
        assert d.item() == pytest.approx(0.5)
        assert i.item() == pytest.approx(1 / 3)

    def test_empty_conventions(self):
        empty, one = _b([[[[0, 0]]]]), _b([[[[1, 0]]]])
        assert dice_iou_per_case(empty, empty)[0].item() == 1
        assert dice_iou_per_case(one, empty)[0].item() == 0  # false positive on empty GT
        assert dice_iou_per_case(empty, one)[0].item() == 0

    def test_per_case_not_pooled(self):
        """Two cases: one perfect, one empty-vs-tiny FP. Mean must be 0.5, not pooled."""
        pred = torch.zeros(2, 1, 10, 10, dtype=torch.bool)
        tgt = torch.zeros(2, 1, 10, 10, dtype=torch.bool)
        pred[0, 0, :5] = tgt[0, 0, :5] = True
        pred[1, 0, 0, 0] = True
        d, _ = dice_iou_per_case(pred, tgt)
        assert d.squeeze(1).tolist() == [1.0, 0.0]

    def test_hd95_identical_is_zero(self):
        m = np.zeros((20, 20, 20), bool)
        m[5:10, 5:10, 5:10] = True
        assert hd95(m, m) == 0.0

    def test_hd95_shift(self):
        a = np.zeros((30, 30), bool)
        a[10:15, 10:15] = True
        b = np.roll(a, 3, axis=0)
        assert hd95(a, b) == pytest.approx(3.0)

    def test_hd95_spacing(self):
        a = np.zeros((30, 30), bool)
        a[10:15, 10:15] = True
        b = np.roll(a, 3, axis=0)
        assert hd95(a, b, spacing=(2.0, 1.0)) == pytest.approx(6.0)

    def test_hd95_empty(self):
        e, m = np.zeros((5, 5), bool), np.eye(5, dtype=bool)
        assert hd95(e, e) == 0.0
        assert hd95(m, e) == HD95_EMPTY_PENALTY

    def test_case_metrics_shapes(self):
        p = torch.rand(2, 3, 8, 8, 8) > 0.5
        t = torch.rand(2, 3, 8, 8, 8) > 0.5
        out = case_metrics(p, t)
        assert set(out) == {"dice", "iou", "hd95"}
        assert all(v.shape == (2, 3) for v in out.values())

    def test_summarize(self):
        s = summarize(torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.5, 1.0]]))
        assert s["mean"].tolist() == pytest.approx([0.5, 1 / 3])
        assert s["median"].tolist() == [0.5, 0.0]
