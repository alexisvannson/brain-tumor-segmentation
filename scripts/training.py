import argparse
import os
import sys
import time
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# Add the project root directory to the Python module search path to enable imports from sibling directories.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Unet import Unet
from models.Unet3d import Unet3d
from datasets.brats_dataset import BraTSDataset
from datasets.brats_dataset3d import BraTSDataset3D


def load_config(model_name):
    """Load base config, optionally merged with model-specific config."""
    base_path = os.path.join("configs", "config.yaml")
    config = {}
    if os.path.exists(base_path):
        with open(base_path) as f:
            config = yaml.safe_load(f) or {}

    model_path = os.path.join("configs", f"{model_name.lower()}.yaml")
    if os.path.exists(model_path):
        with open(model_path) as f:
            config.update(yaml.safe_load(f) or {})

    if not config:
        raise FileNotFoundError(f"No config found for model: {model_name}")

    return config


def create_model(model_name, config):
    """Create Unet or Unet3d instance from model config."""
    model_cfg = config.get("model", {})
    down_layers = [tuple(l) for l in model_cfg["down_layers"]]
    up_layers   = [tuple(l) for l in model_cfg["up_layers"]]
    final_block = tuple(model_cfg["final_block"])
    if model_name.lower() == "unet3d":
        return Unet3d(down_layers, up_layers, final_block)
    return Unet(down_layers, up_layers, final_block)


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    spatial_dims = tuple(range(2, pred.dim()))  # (2,3) for 2D, (2,3,4) for 3D
    intersection = (pred * target).sum(dim=spatial_dims)
    union = pred.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def _crop_masks(masks, preds):
    """Center-crop masks to match preds spatial size (no-padding UNet). Works for 2D and 3D."""
    if preds.shape[2:] == masks.shape[2:]:
        return masks
    if preds.dim() == 5:  # 3D: (B, C, D, H, W)
        d, h, w = preds.shape[2], preds.shape[3], preds.shape[4]
        dd = (masks.shape[2] - d) // 2
        dh = (masks.shape[3] - h) // 2
        dw = (masks.shape[4] - w) // 2
        return masks[:, :, dd:dd+d, dh:dh+h, dw:dw+w]
    # 2D: (B, C, H, W)
    h, w = preds.shape[2], preds.shape[3]
    dh = (masks.shape[2] - h) // 2
    dw = (masks.shape[3] - w) // 2
    return masks[:, :, dh:dh+h, dw:dw+w]


def validate(model, val_loader, criterion, device):
    """
    Validate the model on validation set.

    Returns:
        float: avg combined loss (BCE + Dice)
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            masks = _crop_masks(masks, preds)
            bce  = criterion(preds, masks)
            dice = dice_loss(preds, masks)
            total_loss += (bce + dice).item()

    return total_loss / len(val_loader)


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs,
    val_loader=None,
    patience=5,
    scheduler=None,
    output_path="weights",
    weights_name="final_model",
    start_weights=None,
    use_wandb=False,
):
    best_val_loss = float("inf")
    patience_counter = 0

    if start_weights:
        model.load_state_dict(torch.load(start_weights, map_location=device))

    os.makedirs(output_path, exist_ok=True)
    print(f"Training model in {output_path}")

    start_time = datetime.now()
    log_path = os.path.join(output_path, f"training_logs_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_path, "w") as f:
        f.write(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epochs: {epochs}, Patience: {patience}\n")
        f.write(f"Output path: {output_path}\n")
        f.write(f"Device: {device}\n")
        f.write("-" * 50 + "\n")

    print("start training")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        model.train()

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            masks = _crop_masks(masks, preds)
            bce  = criterion(preds, masks)
            dice = dice_loss(preds, masks)
            loss = bce + dice
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - t0

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            if scheduler is not None:
                scheduler.step(val_loss)
            log_line = (f"Epoch {epoch:03d}/{epochs} | train_loss={avg_train_loss:.4f} |"
                        f" val_loss={val_loss:.4f}")
            if use_wandb:
                wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss})
        else:
            val_loss = avg_train_loss
            log_line = f"Epoch {epoch:03d}/{epochs} | train_loss={avg_train_loss:.4f}"
            if use_wandb:
                wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")
            f.write(f"Epoch {epoch}/{epochs} took {epoch_time / 60:.2f} minutes\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(output_path, "best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (val_loss={val_loss:.4f}): {best_model_path}")
        else:
            patience_counter += 1

        print(f"Epoch {epoch} took {epoch_time:.2f} seconds")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    final_model_path = os.path.join(output_path, f"{weights_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    with open(log_path, "a") as f:
        f.write("-" * 50 + "\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Best val_loss achieved: {best_val_loss:.4f}\n")
        f.write(f"Final model saved: {final_model_path}\n")


def resolve_path(paths, writable=False):
    """Return the first valid path from a list, or the string as-is."""
    if not isinstance(paths, list):
        return paths
    for path in paths:
        check = os.path.dirname(path) if writable and os.path.dirname(path) else path
        if os.path.exists(check) and (not writable or os.access(check, os.W_OK)):
            return path
    return paths[0]


def setup_dataloaders(config, model_name="unet"):
    """Build train and validation DataLoaders from config."""
    training_cfg = config.get("training", {})
    if model_name.lower() == "unet3d":
        dataset = BraTSDataset3D(training_cfg["data_dir"])
    else:
        dataset = BraTSDataset(training_cfg["data_dir"])
    val_size   = int(len(dataset) * training_cfg.get("val_split", 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    batch_size  = training_cfg.get("batch_size", 2)
    num_workers = training_cfg.get("num_workers", 0)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def setup_optimizer(model, train_config):
    """Create optimizer from config."""
    optimizer_type = train_config.get("optimizer", "Adam")
    lr = train_config.get("lr", train_config.get("learning_rate", 1e-4))
    weight_decay = train_config.get("weight_decay", 0.0)

    if optimizer_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                         momentum=train_config.get("momentum", 0.9), nesterov=True)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


def setup_scheduler(optimizer, train_config):
    """Create LR scheduler from config, or return None if not configured."""
    sched_config = train_config.get("scheduler")
    if not sched_config:
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    name = sched_config.get("name", "ReduceLROnPlateau")
    if name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=sched_config.get("factor", 0.5),
            patience=sched_config.get("patience", 5),
            min_lr=sched_config.get("min_lr", 1e-6),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def setup_criterion(config):
    """Instantiate loss function (BCEWithLogitsLoss for segmentation)."""
    return nn.BCEWithLogitsLoss()


def main():
    parser = argparse.ArgumentParser(description="Train model script")
    parser.add_argument("model", type=str, help="Model name to train")
    parser.add_argument("--config", type=str, default=None, help="Override default config file path")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights for fine-tuning")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(args.model, config).to(device)

    train_loader, val_loader = setup_dataloaders(config, model_name=args.model)

    train_config = config.get("training", {})
    out_path_cfg = train_config.get("output_paths") or train_config.get("checkpoint_dir", "models/checkpoints")
    output_path = resolve_path(out_path_cfg, writable=True)

    # Init W&B if available
    use_wandb = False
    if wandb is not None:
        try:
            wandb.init(project="brats", name=args.model, config=config)
            use_wandb = True
            print("W&B logging enabled.")
        except Exception as e:
            print(f"W&B not available, skipping: {e}")

    optimizer = setup_optimizer(model, train_config)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=setup_criterion(config),
        optimizer=optimizer,
        scheduler=setup_scheduler(optimizer, train_config),
        device=device,
        output_path=output_path,
        weights_name=args.model,
        epochs=train_config.get("epochs", 50),
        patience=train_config.get("patience", 10),
        start_weights=args.weights,
        use_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
