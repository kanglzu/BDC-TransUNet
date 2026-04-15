# Dataset Preparation

BDC-TransUNet requires datasets to be preprocessed into `.npz` format with `image` and `label` keys.

## Supported Datasets

### GlaS (Gland Segmentation Challenge)

- **Paper**: K. Sirinukunwattana et al., "Gland segmentation in colon histology images: The GlaS challenge contest," *Medical Image Analysis*, vol. 35, pp. 489–502, 2017.
- **Download**: https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation
- **Task**: Gland segmentation in H&E stained histology images
- **Split**: 115 train / 24 val / 26 test

### CVC-ClinicDB

- **Paper**: J. Bernal et al., "WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians," *Computerized Medical Imaging and Graphics*, vol. 43, pp. 99–111, 2015.
- **Download**: https://www.kaggle.com/datasets/balraj98/cvcclinicdb
- **Task**: Polyp segmentation in colonoscopy images
- **Split**: 489 train / 62 val / 61 test (80%/10%/10%)

### Kvasir-SEG

- **Paper**: D. Jha et al., "Kvasir-SEG: A segmented polyp dataset," in *International Conference on Multimedia Modeling*, Springer, 2020, pp. 451–462.
- **Download**: https://datasets.simula.no/kvasir-seg/
- **Task**: Polyp segmentation in colonoscopy images
- **Split**: 700 train / 200 val / 100 test

## Expected Directory Structure

After downloading and preprocessing, organize the data as follows:

```
data/
├── GLAS/
│   ├── train_npz/
│   │   ├── sample_001.npz
│   │   ├── sample_002.npz
│   │   └── ...
│   ├── val_npz/
│   │   └── ...
│   └── test_npz/
│       └── ...
├── Kvasir/
│   ├── train_npz/
│   ├── val_npz/
│   └── test_npz/
└── CVC/
    ├── train_npz/
    ├── val_npz/
    └── test_npz/
```

## NPZ File Format

Each `.npz` file should contain:

- `image`: numpy array of shape `(H, W, 3)`, dtype `float32`, range `[0, 1]`
- `label`: numpy array of shape `(H, W)`, dtype `uint8`, values `{0, 1}`

## Preprocessing Script

You can convert raw images and masks to `.npz` format with:

```python
import numpy as np
from PIL import Image

img = np.array(Image.open("image.png").convert("RGB")).astype(np.float32) / 255.0
mask = (np.array(Image.open("mask.png").convert("L")) > 128).astype(np.uint8)
np.savez("sample.npz", image=img, label=mask)
```
