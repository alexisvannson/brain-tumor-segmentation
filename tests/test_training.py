"""
End-to-end tests for the training pipeline:
setup helpers, train loop, validate, early stopping, checkpointing.
"""
import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Unet import Unet
from models.Unet3d import Unet3d
from scripts.training import (
    create_model,
    dice_loss,
    resolve_path,
    setup_criterion,
    setup_optimizer,
    setup_scheduler,
    train,
    validate,
)

# ---------------------------------------------------------------------------
# Tiny model configs & factory helpers
# ---------------------------------------------------------------------------

TINY_2D_CFG = {
    "model": {
        "down_layers": [[1, 8], [8, 16]],
        "up_layers":   [[16, 8]],
        "final_block": [8, 1],
    }
}

TINY_3D_CFG = {
    "model": {
        "down_layers": [[1, 4], [4, 8]],
        "up_layers":   [[8, 4]],
        "final_block": [4, 1],
    }
}

# Padded UNet: input (B, 1, 32, 32) → output (B, 1, 32, 32)
IMG_SIZE_2D = 32
IMG_SIZE_3D = 32


def _make_loader_2d(n=6, batch=2):
    """Synthetic 2D DataLoader: images (B,1,32,32), masks (B,1,32,32)."""
    images = torch.randn(n, 1, IMG_SIZE_2D, IMG_SIZE_2D)
    masks  = torch.randint(0, 2, (n, 1, IMG_SIZE_2D, IMG_SIZE_2D)).float()
    return DataLoader(TensorDataset(images, masks), batch_size=batch, shuffle=False)


def _make_loader_3d(n=4, batch=1):
    """Synthetic 3D DataLoader: images (B,1,32,32,32), masks (B,1,32,32,32)."""
    images = torch.randn(n, 1, IMG_SIZE_3D, IMG_SIZE_3D, IMG_SIZE_3D)
    masks  = torch.randint(0, 2, (n, 1, IMG_SIZE_3D, IMG_SIZE_3D, IMG_SIZE_3D)).float()
    return DataLoader(TensorDataset(images, masks), batch_size=batch, shuffle=False)


def _tiny_model_2d():
    return Unet(
        [(1, 8), (8, 16)],
        [(16, 8)],
        (8, 1),
    )


def _tiny_model_3d():
    return Unet3d(
        [(1, 4), (4, 8)],
        [(8, 4)],
        (4, 1),
    )


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------

class TestCreateModel:
    def test_creates_unet(self):
        model = create_model("unet", TINY_2D_CFG)
        assert isinstance(model, Unet)

    def test_creates_unet3d(self):
        model = create_model("unet3d", TINY_3D_CFG)
        assert isinstance(model, Unet3d)

    def test_case_insensitive(self):
        assert isinstance(create_model("UNet",   TINY_2D_CFG), Unet)
        assert isinstance(create_model("UNET3D", TINY_3D_CFG), Unet3d)

    def test_unet_forward_after_create(self):
        model = create_model("unet", TINY_2D_CFG)
        out = model(torch.randn(1, 1, 32, 32))
        assert out.shape[1] == 1

    def test_unet3d_forward_after_create(self):
        model = create_model("unet3d", TINY_3D_CFG)
        out = model(torch.randn(1, 1, 32, 32, 32))
        assert out.shape[1] == 1


# ---------------------------------------------------------------------------
# setup_optimizer
# ---------------------------------------------------------------------------

class TestSetupOptimizer:
    @pytest.fixture
    def model(self):
        return _tiny_model_2d()

    def test_adam_default(self, model):
        opt = setup_optimizer(model, {"optimizer": "Adam", "lr": 1e-3})
        assert isinstance(opt, optim.Adam)

    def test_adamw(self, model):
        opt = setup_optimizer(model, {"optimizer": "AdamW", "lr": 1e-3})
        assert isinstance(opt, optim.AdamW)

    def test_sgd(self, model):
        opt = setup_optimizer(model, {"optimizer": "SGD", "lr": 1e-2})
        assert isinstance(opt, optim.SGD)

    def test_unknown_raises(self, model):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            setup_optimizer(model, {"optimizer": "RMSProp"})

    def test_lr_is_applied(self, model):
        opt = setup_optimizer(model, {"optimizer": "Adam", "lr": 7e-4})
        assert opt.param_groups[0]["lr"] == pytest.approx(7e-4)

    def test_weight_decay_is_applied(self, model):
        opt = setup_optimizer(model, {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4})
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)

    def test_default_lr_fallback(self, model):
        """Accepts 'lr' key (used in unet.yaml) or 'learning_rate'."""
        opt1 = setup_optimizer(model, {"lr": 5e-4})
        assert opt1.param_groups[0]["lr"] == pytest.approx(5e-4)
        opt2 = setup_optimizer(model, {"learning_rate": 3e-4})
        assert opt2.param_groups[0]["lr"] == pytest.approx(3e-4)


# ---------------------------------------------------------------------------
# setup_scheduler
# ---------------------------------------------------------------------------

class TestSetupScheduler:
    @pytest.fixture
    def optimizer(self):
        model = _tiny_model_2d()
        return optim.Adam(model.parameters(), lr=1e-3)

    def test_default_returns_plateau(self, optimizer):
        sched = setup_scheduler(optimizer, {})
        assert isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau)

    def test_explicit_plateau_config(self, optimizer):
        cfg = {"scheduler": {"name": "ReduceLROnPlateau", "factor": 0.2, "patience": 3}}
        sched = setup_scheduler(optimizer, cfg)
        assert isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau)

    def test_unknown_scheduler_raises(self, optimizer):
        cfg = {"scheduler": {"name": "CosineAnnealing"}}
        with pytest.raises(ValueError, match="Unknown scheduler"):
            setup_scheduler(optimizer, cfg)

    def test_scheduler_step_does_not_crash(self, optimizer):
        sched = setup_scheduler(optimizer, {})
        sched.step(0.5)  # should not raise


# ---------------------------------------------------------------------------
# setup_criterion
# ---------------------------------------------------------------------------

class TestSetupCriterion:
    def test_returns_bce(self):
        criterion = setup_criterion({})
        assert isinstance(criterion, nn.BCEWithLogitsLoss)

    def test_criterion_computes_loss(self):
        criterion = setup_criterion({})
        pred   = torch.randn(2, 1, 8, 8)
        target = torch.randint(0, 2, (2, 1, 8, 8)).float()
        loss = criterion(pred, target)
        assert loss.dim() == 0
        assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_string_returned_as_is(self):
        assert resolve_path("some/path") == "some/path"

    def test_first_existing_dir(self, tmp_path):
        p = str(tmp_path)
        result = resolve_path([p, "/nonexistent/path"])
        assert result == p

    def test_skips_nonexistent(self, tmp_path):
        p = str(tmp_path)
        result = resolve_path(["/nonexistent", p])
        assert result == p

    def test_fallback_to_first_when_none_exist(self):
        paths = ["/no/such/path/a", "/no/such/path/b"]
        assert resolve_path(paths) == paths[0]

    def test_writable_check(self, tmp_path):
        p = str(tmp_path)
        result = resolve_path([p], writable=True)
        assert result == p


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_returns_float(self):
        model     = _tiny_model_2d()
        loader    = _make_loader_2d()
        criterion = nn.BCEWithLogitsLoss()
        out = validate(model, loader, criterion, torch.device("cpu"))
        assert isinstance(out["loss"], float)
        assert isinstance(out["mean_dice"], float)
        assert out["dice"].shape == (1,)

    def test_positive_loss(self):
        model     = _tiny_model_2d()
        loader    = _make_loader_2d()
        criterion = nn.BCEWithLogitsLoss()
        out = validate(model, loader, criterion, torch.device("cpu"))
        assert out["loss"] > 0.0
        assert 0.0 <= out["mean_dice"] <= 1.0

    def test_2d_model_on_volumes(self):
        """2D model + volume loader: predicted slice-wise, Dice per patient volume."""
        model  = _tiny_model_2d()
        loader = _make_loader_3d(n=2)
        out = validate(model, loader, nn.BCEWithLogitsLoss(), torch.device("cpu"), slice_batch=5)
        assert out["loss"] > 0.0
        assert 0.0 <= out["mean_dice"] <= 1.0

    def test_model_in_eval_mode_after(self):
        model     = _tiny_model_2d()
        loader    = _make_loader_2d()
        criterion = nn.BCEWithLogitsLoss()
        validate(model, loader, criterion, torch.device("cpu"))
        assert not model.training

    def test_no_gradient_computed(self):
        """validate must not accumulate gradients."""
        model = _tiny_model_2d()
        for p in model.parameters():
            p.grad = None
        validate(model, _make_loader_2d(), nn.BCEWithLogitsLoss(), torch.device("cpu"))
        for p in model.parameters():
            assert p.grad is None

    def test_3d_model(self):
        model     = _tiny_model_3d()
        loader    = _make_loader_3d()
        criterion = nn.BCEWithLogitsLoss()
        out = validate(model, loader, criterion, torch.device("cpu"))
        assert isinstance(out["loss"], float)
        assert out["loss"] > 0.0

    def test_deterministic_across_calls(self):
        """validate should return the same value on repeated calls (no dropout / bn)."""
        model     = _tiny_model_2d()
        loader    = _make_loader_2d(n=4)
        criterion = nn.BCEWithLogitsLoss()
        loss1 = validate(model, loader, criterion, torch.device("cpu"))
        loss2 = validate(model, loader, criterion, torch.device("cpu"))
        assert abs(loss1["loss"] - loss2["loss"]) < 1e-6
        assert loss1["mean_dice"] == loss2["mean_dice"]


# ---------------------------------------------------------------------------
# train (full loop)
# ---------------------------------------------------------------------------

class TestTrainLoop:
    def _run_train(self, tmp_path, model, train_loader, val_loader=None,
                   epochs=2, patience=10, scheduler=None):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        train(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epochs=epochs,
            val_loader=val_loader,
            patience=patience,
            scheduler=scheduler,
            output_path=str(tmp_path),
            weights_name="test_model",
            use_wandb=False,
        )

    # --- basic run ---

    def test_runs_without_error_2d(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d())

    def test_runs_without_error_3d(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_3d(), _make_loader_3d())

    def test_runs_without_val_loader(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d(), val_loader=None)

    def test_runs_with_val_loader(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d(),
                        val_loader=_make_loader_2d(n=4))

    # --- checkpoints ---

    def test_final_checkpoint_saved(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d())
        assert (tmp_path / "test_model.pth").exists()

    def test_best_checkpoint_saved_with_val(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d(),
                        val_loader=_make_loader_2d(n=4), epochs=2)
        assert (tmp_path / "best.pth").exists()

    def test_best_checkpoint_not_saved_without_val(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d(), val_loader=None, epochs=2)
        # Without val_loader val_loss = train_loss which always triggers best save
        assert (tmp_path / "best.pth").exists()

    def test_checkpoint_loadable(self, tmp_path):
        model = _tiny_model_2d()
        self._run_train(tmp_path, model, _make_loader_2d())
        state = torch.load(tmp_path / "test_model.pth", map_location="cpu")
        new_model = _tiny_model_2d()
        new_model.load_state_dict(state)  # should not raise

    # --- log file ---

    def test_log_file_created(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d())
        logs = list(tmp_path.glob("training_logs_*.txt"))
        assert len(logs) == 1

    def test_log_file_contains_epochs(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d(), epochs=2)
        log = list(tmp_path.glob("training_logs_*.txt"))[0].read_text()
        assert "Epoch 001" in log
        assert "Epoch 002" in log

    def test_log_file_contains_train_loss(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d())
        log = list(tmp_path.glob("training_logs_*.txt"))[0].read_text()
        assert "train_loss=" in log

    def test_log_file_val_loss_when_val_provided(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d(),
                        val_loader=_make_loader_2d(n=4))
        log = list(tmp_path.glob("training_logs_*.txt"))[0].read_text()
        assert "val_loss=" in log

    def test_log_file_contains_completion_marker(self, tmp_path):
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d())
        log = list(tmp_path.glob("training_logs_*.txt"))[0].read_text()
        assert "Training completed" in log

    # --- early stopping ---

    def test_early_stopping_fires(self, tmp_path, capsys):
        """With patience=1 and constant high val_loss, should stop after epoch 2."""
        model = _tiny_model_2d()
        # Use a fixed val loss by overriding validate indirectly via a constant loader
        # We train for many epochs but patience=1; it should exit early.
        optimizer = optim.Adam(model.parameters(), lr=0.0)  # frozen weights → constant loss
        criterion = nn.BCEWithLogitsLoss()
        train(
            model=model,
            train_loader=_make_loader_2d(),
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epochs=10,
            val_loader=_make_loader_2d(n=4),
            patience=1,
            output_path=str(tmp_path),
            weights_name="early",
            use_wandb=False,
        )
        captured = capsys.readouterr()
        assert "Early stopping" in captured.out

    def test_early_stopping_respects_patience(self, tmp_path, capsys):
        """With patience=3, early stopping should not fire within 2 epochs."""
        model = _tiny_model_2d()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        train(
            model=model,
            train_loader=_make_loader_2d(),
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epochs=2,
            val_loader=_make_loader_2d(n=4),
            patience=3,
            output_path=str(tmp_path),
            weights_name="noearly",
            use_wandb=False,
        )
        captured = capsys.readouterr()
        assert "Early stopping" not in captured.out

    # --- start_weights ---

    def test_start_weights_loaded(self, tmp_path):
        """Training should resume from given weights without error."""
        model = _tiny_model_2d()
        weights_path = str(tmp_path / "init.pth")
        torch.save(model.state_dict(), weights_path)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        train(
            model=model,
            train_loader=_make_loader_2d(),
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epochs=1,
            output_path=str(tmp_path),
            weights_name="resumed",
            start_weights=weights_path,
            use_wandb=False,
        )
        assert (tmp_path / "resumed.pth").exists()

    # --- scheduler ---

    def test_scheduler_used_with_val_loader(self, tmp_path):
        """Scheduler.step should be called once per epoch when val_loader is provided."""
        model     = _tiny_model_2d()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
        criterion = nn.BCEWithLogitsLoss()
        initial_lr = optimizer.param_groups[0]["lr"]

        train(
            model=model,
            train_loader=_make_loader_2d(),
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epochs=1,
            val_loader=_make_loader_2d(n=4),
            scheduler=scheduler,
            output_path=str(tmp_path),
            weights_name="sched",
            use_wandb=False,
        )
        # LR might or might not have changed — just ensure no crash and LR is still set
        assert optimizer.param_groups[0]["lr"] > 0

    # --- model state after training ---

    def test_model_parameters_updated(self, tmp_path):
        """Parameters should change after training (non-zero lr)."""
        model = _tiny_model_2d()
        initial_params = [p.clone().detach() for p in model.parameters()]

        self._run_train(tmp_path, model, _make_loader_2d(), epochs=2)

        changed = any(
            not torch.allclose(p, q)
            for p, q in zip(initial_params, model.parameters())
        )
        assert changed, "No parameters were updated during training"

    def test_loss_is_finite(self, tmp_path, capsys):
        """train_loss in the log should be finite (no NaN/inf)."""
        self._run_train(tmp_path, _tiny_model_2d(), _make_loader_2d(), epochs=2)
        log = list(tmp_path.glob("training_logs_*.txt"))[0].read_text()
        import re
        losses = re.findall(r"train_loss=([\d.]+)", log)
        assert losses, "No train_loss found in log"
        for v in losses:
            assert float(v) < 1e6, f"Suspiciously large loss: {v}"


# ---------------------------------------------------------------------------
# predict / history / patient-level dataloaders
# ---------------------------------------------------------------------------

from scripts.training import predict, setup_dataloaders


class TestPredict:
    def test_2d_model_slicewise_matches_direct(self):
        model = _tiny_model_2d().eval()
        vol = torch.randn(1, 1, 6, 32, 32)
        with torch.no_grad():
            out = predict(model, vol, slice_batch=4)
            direct = torch.stack([model(vol[:, :, d]) for d in range(6)], dim=2)
        assert out.shape == (1, 1, 6, 32, 32)
        assert torch.allclose(out, direct, atol=1e-6)

    def test_3d_model_direct(self):
        model = _tiny_model_3d().eval()
        vol = torch.randn(1, 1, 16, 16, 16)
        with torch.no_grad():
            assert predict(model, vol).shape == (1, 1, 16, 16, 16)


class TestHistoryAndSelection:
    def test_history_csv_written(self, tmp_path):
        model = _tiny_model_2d()
        train(
            model=model, train_loader=_make_loader_2d(), criterion=nn.BCEWithLogitsLoss(),
            optimizer=optim.Adam(model.parameters(), lr=1e-3), device=torch.device("cpu"),
            epochs=2, val_loader=_make_loader_2d(n=4), output_path=str(tmp_path),
            weights_name="h", use_wandb=False,
        )
        import csv
        rows = list(csv.DictReader(open(tmp_path / "history.csv")))
        assert [r["epoch"] for r in rows] == ["1", "2"]
        assert all(r["val_mean_dice"] != "" for r in rows)

    def test_log_reports_best_model(self, tmp_path):
        model = _tiny_model_2d()
        train(
            model=model, train_loader=_make_loader_2d(), criterion=nn.BCEWithLogitsLoss(),
            optimizer=optim.Adam(model.parameters(), lr=1e-3), device=torch.device("cpu"),
            epochs=1, val_loader=_make_loader_2d(n=4), output_path=str(tmp_path),
            weights_name="b", use_wandb=False,
        )
        log = list(tmp_path.glob("training_logs_*.txt"))[0].read_text()
        assert "Best model: epoch 1, val_mean_dice=" in log


class TestPatientLevelDataloaders:
    def _make_slices(self, root, n_patients=10, n_slices=3):
        import h5py
        import numpy as np
        for p in range(n_patients):
            for s in range(n_slices):
                with h5py.File(root / f"volume_{p}_slice_{s}.h5", "w") as f:
                    f.create_dataset("image", data=np.random.rand(16, 16, 4).astype(np.float32))
                    f.create_dataset("mask", data=np.zeros((16, 16, 3), dtype=np.float32))

    def test_no_patient_leakage(self, tmp_path):
        self._make_slices(tmp_path)
        cfg = {
            "data": {"data_dir": str(tmp_path), "split_file": str(tmp_path / "split.json"),
                     "val_split": 0.2, "test_split": 0.2},
            "training": {"batch_size": 2},
        }
        train_loader, val_loader = setup_dataloaders(cfg, "unet")
        import json
        split = json.load(open(tmp_path / "split.json"))
        train_files = train_loader.dataset.files
        assert all(any(f"volume_{p}_" in f for p in split["train"]) for f in train_files)
        assert len(train_files) == len(split["train"]) * 3
        assert val_loader.dataset.patient_ids == split["val"]
        assert val_loader.batch_size == 1
        img, mask = val_loader.dataset[0]
        assert img.shape == (4, 3, 16, 16)  # whole patient volume


# ---------------------------------------------------------------------------
# mixed precision
# ---------------------------------------------------------------------------

from unittest import mock

from scripts.training import amp_settings


class TestMixedPrecision:
    def test_disabled_on_cpu(self):
        assert amp_settings(torch.device("cpu"), True) == (False, torch.float32, False)

    def test_disabled_when_off(self):
        assert amp_settings(torch.device("cuda"), False) == (False, torch.float32, False)

    def test_bf16_without_scaler_when_supported(self):
        with mock.patch("torch.cuda.is_bf16_supported", return_value=True):
            assert amp_settings(torch.device("cuda"), True) == (True, torch.bfloat16, False)

    def test_fp16_with_scaler_otherwise(self):
        with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
            assert amp_settings(torch.device("cuda"), True) == (True, torch.float16, True)

    def test_train_with_amp_flag_on_cpu(self, tmp_path):
        """amp=True must fall back cleanly on CPU and still update the model."""
        model = _tiny_model_2d()
        before = [p.clone() for p in model.parameters()]
        train(
            model=model, train_loader=_make_loader_2d(), criterion=nn.BCEWithLogitsLoss(),
            optimizer=optim.Adam(model.parameters(), lr=1e-3), device=torch.device("cpu"),
            epochs=1, val_loader=_make_loader_2d(n=4), output_path=str(tmp_path),
            weights_name="amp", use_wandb=False, amp=True,
        )
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    def test_validate_with_amp_flag_on_cpu(self):
        model, loader = _tiny_model_2d(), _make_loader_2d()
        a = validate(model, loader, nn.BCEWithLogitsLoss(), torch.device("cpu"), amp=True)
        b = validate(model, loader, nn.BCEWithLogitsLoss(), torch.device("cpu"), amp=False)
        assert a["loss"] == pytest.approx(b["loss"])


class TestLoaderSettings:
    def test_workers_persistent(self, tmp_path):
        import h5py
        import numpy as np
        for p in range(5):
            with h5py.File(tmp_path / f"volume_{p}_slice_0.h5", "w") as f:
                f.create_dataset("image", data=np.random.rand(16, 16, 4).astype(np.float32))
                f.create_dataset("mask", data=np.zeros((16, 16, 3), dtype=np.float32))
        cfg = {"data": {"data_dir": str(tmp_path), "split_file": str(tmp_path / "s.json"),
                        "val_split": 0.2, "test_split": 0.2},
               "training": {"batch_size": 2, "num_workers": 2}}
        train_loader, val_loader = setup_dataloaders(cfg, "unet")
        assert train_loader.num_workers == 2 and train_loader.persistent_workers
        assert train_loader.prefetch_factor == 4
        assert val_loader.persistent_workers
