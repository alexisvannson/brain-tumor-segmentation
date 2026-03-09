# Brain Tumor Segmentation

Segmentation of brain tumors from multimodal MRI using U-Net architectures trained on the BraTS 2020 dataset. Supports both 2D slice-based and full 3D volumetric training.

## Project Structure

```
├── models/
│   ├── Unet.py          # 2D U-Net
│   └── Unet3d.py        # 3D U-Net
├── datasets/
│   ├── brats_dataset.py    # 2D slice loader (H5)
│   └── brats_dataset3d.py  # 3D volume loader (H5)
├── scripts/
│   ├── training.py      # Training loop, validation, checkpointing
│   └── evaluate.py      # Evaluation: Dice and IoU per class
├── configs/
│   ├── unet.yaml         # 2D U-Net config
│   └── unet3d.yaml       # 3D U-Net config
├── tests/               # Pytest test suite (121 tests)
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

Set `data_dir` in the relevant config to point to your H5 files.

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

Checkpoints are saved to `checkpoint_dir` (set in config). `best.pth` is updated whenever validation loss improves.

## Evaluation

```bash
# Evaluate 2D U-Net on full dataset
python scripts/evaluate.py unet checkpoints/best.pth

# Evaluate 3D U-Net with a custom data directory
python scripts/evaluate.py unet3d checkpoints/best.pth --data_dir /path/to/data

# Save results to a file
python scripts/evaluate.py unet checkpoints/best.pth --output results/eval.txt
```

Reports per-class and mean Dice and IoU for the three BraTS regions (whole tumor, tumor core, enhancing tumor).

## Tests

```bash
pytest tests/ -v
```

121 tests covering model forward passes, dataset loading, loss functions, and the full training loop.

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
