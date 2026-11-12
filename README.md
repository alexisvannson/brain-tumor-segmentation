# Brain Tumor Segmentation

Segmentation of brain tumors from multimodal MRI using U-Net architectures trained on the BraTS 2020 dataset. Supports both 2D slice-based and full 3D volumetric training.

## Project Structure

```
├── models/
│   ├── Unet.py          # 2D U-Net
│   └── Unet3d.py        # 3D U-Net
├── datasets/
│   ├── brats_dataset.py    # 2D slice loader (H5)
│   ├── brats_dataset3d.py  # 3D volume loader (stacks slice files per patient)
│   ├── splits.py           # Patient-level train/val/test split
│   └── transforms.py       # Label -> region conversion, z-score normalisation
├── scripts/
│   ├── training.py      # Training loop, validation, checkpointing
│   ├── evaluate.py      # Per-patient Dice, IoU, HD95 on the held-out test split
│   ├── metrics.py       # Metric implementations (BraTS conventions)
│   └── check_masks.py   # Check whether masks are one-hot labels or nested regions
├── configs/
│   ├── unet.yaml         # 2D U-Net config
│   └── unet3d.yaml       # 3D U-Net config
├── notebooks/
│   └── colab_train_eval.ipynb  # Colab: train, evaluate, plot
├── tests/               # Pytest test suite
```

## Models

Both models follow the original U-Net encoder-decoder structure with skip connections and center-crop concatenation.

**2D U-Net** — operates on individual axial slices `(4, H, W)`:
- 5-level encoder: 4 → 64 → 128 → 256 → 512 → 1024 channels
- Loss: BCE + Dice

**3D U-Net** — operates on full volumes `(4, D, H, W)`:
- 5-level encoder: 4 → 32 → 64 → 128 → 256 → 320 channels (reduced for memory)
- Loss: BCE + Dice

Output: 3-channel segmentation mask (whole tumor, tumor core, enhancing tumor).

## Data

The models expect BraTS 2020 H5 files. Each file contains:
- `image`: 4 MRI modalities (T1, T1ce, T2, FLAIR)
- `mask`: 3 binary segmentation channels

Set `data.data_dir` in the relevant config to point to your H5 files.

**Patient-level split.** Slices of one patient are highly correlated, so splitting by file leaks
information into validation. On the first training run, patients (not slices) are split into
train / val / test with a fixed seed and saved to `data.split_file`
(default `splits/brats2020_seed42.json`). Later runs and `evaluate.py` reuse this file. The test patients
are never used for training or model selection. Commit the split file so results are reproducible.

**Labels.** Models are trained and evaluated on the BraTS regions: whole tumor (NCR ∪ ED ∪ ET),
tumor core (NCR ∪ ET) and enhancing tumor (ET). With `mask_format: labels` the raw one-hot NCR / ED / ET
channels are converted to these regions. Run `python scripts/check_masks.py <data_dir>` to confirm which
format your files use.

**Normalisation.** Each modality is z-scored over brain voxels (non-zero), per slice for the 2D model
and per volume for the 3D model (`data.normalize`). Evaluation uses the same normalisation as training.

## Training

```bash
# 2D U-Net
python scripts/training.py unet

# 3D U-Net
python scripts/training.py unet3d

# Resume from checkpoint
python scripts/training.py unet --weights checkpoints/best.pth

# Custom config
python scripts/training.py unet --config configs/unet.yaml
```

Checkpoints are saved to `checkpoint_dir` (set in config). Validation runs on whole patient volumes
(the 2D model predicts slice by slice). `best.pth` is the checkpoint with the highest validation mean Dice,
and early stopping and the LR scheduler track the same score. Each run also writes `history.csv`
(per-epoch losses, per-region val Dice, learning rate) and a copy of the config it used.

Both models use padded convolutions by default (`model.padding: 1`), so predictions cover the
full image. `padding: 0` restores the original valid-convolution U-Net, whose output is only a centre crop.

On a GPU, training uses mixed precision (`training.amp: true`): bfloat16 on Ampere or newer (e.g. A100), and
float16 with loss scaling on older GPUs (e.g. T4). The loss is always computed in float32. Set `amp: false`
for full float32.

## Evaluation

```bash
# Evaluate 2D U-Net on the held-out test patients
python scripts/evaluate.py unet checkpoints/best.pth

# Evaluate on the validation patients instead, with a different threshold
python scripts/evaluate.py unet checkpoints/best.pth --split val --threshold 0.4

# Save summary (results/eval.txt) and per-patient metrics (results/eval.csv)
python scripts/evaluate.py unet checkpoints/best.pth --output results/eval.txt
```

Metrics are computed **per patient** on the full 3D volume, then aggregated (mean ± std and median across
patients): Dice, IoU and HD95 (95th-percentile Hausdorff distance, in voxels, which is mm for BraTS) for whole
tumor, tumor core and enhancing tumor. Empty regions follow BraTS conventions: if both prediction and ground
truth are empty, Dice = 1 and HD95 = 0. If only one of them is empty, Dice = 0 and HD95 = 373.13. The
per-patient CSV is the input for confidence intervals and paired statistical tests between models.

## Colab

`notebooks/colab_train_eval.ipynb` runs the whole pipeline on Colab with the dataset on Google Drive:
mask-format check, training, learning curves, threshold tuning on validation patients, a single
held-out test evaluation with bootstrap 95% CIs, per-patient distributions, Dice vs tumour size,
and best / median / worst overlays. Outputs are saved to Drive.

## Tests

```bash
pytest tests/ -v
```

Tests cover model forward passes, dataset loading, patient-level splits, label conversion, metrics, loss functions, and the full training loop.

## References

**Dataset**
BraTS 2020 Training Data  
Kaggle Dataset: [https://www.kaggle.com/datasets/awsaf49/brats2020-training-data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)  


**U-Net**
- O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI 2015*, LNCS vol. 9351, pp. 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

**3D U-Net**
- Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger, "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation," *MICCAI 2016*, LNCS vol. 9901, pp. 424–432. https://doi.org/10.1007/978-3-319-46723-8_49

**Dice Loss**
- F. Milletari, N. Navab, S.-A. Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation," *3DV 2016*, pp. 565–571. https://doi.org/10.1109/3DV.2016.79
