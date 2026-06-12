# Dataset Preparation

BDC-TransUNet requires datasets to be preprocessed into `.npz` format with `image` and `label` keys.

## Supported Datasets

### CVC-ClinicDB

- **Paper**: J. Bernal et al., "WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians," *Computerized Medical Imaging and Graphics*, vol. 43, pp. 99–111, 2015.
- **Download**: https://www.kaggle.com/datasets/balraj98/cvcclinicdb
- **Task**: Polyp segmentation in colonoscopy images
- **Split**: 489 train / 61 val / 62 test (seed=1234, ratio=80%/10%/10%)

### GlaS (Gland Segmentation Challenge)

- **Paper**: K. Sirinukunwattana et al., "Gland segmentation in colon histology images: The GlaS challenge contest," *Medical Image Analysis*, vol. 35, pp. 489–502, 2017.
- **Download**: https://websignon.warwick.ac.uk/origin/slogin?shire=https%3A%2F%2Fwarwick.ac.uk%2Fsitebuilder2%2Fshire-read&providerId=urn%3Awarwick.ac.uk%3Asitebuilder2%3Aread%3Aservice&target=https%3A%2F%2Fwarwick.ac.uk%2Ffac%2Fcross_fac%2Ftia%2Fdata%2Fglascontest&status=notloggedin
- **Task**: Gland segmentation in H&E stained histology images
- **Split**: 115 train / 24 val / 26 test (seed=1234, ratio=70%/15%/15%)

### Kvasir-SEG

- **Paper**: D. Jha et al., "Kvasir-SEG: A segmented polyp dataset," in *International Conference on Multimedia Modeling*, Springer, 2020, pp. 451–462.
- **Download**: https://datasets.simula.no/kvasir-seg/
- **Task**: Polyp segmentation in colonoscopy images
- **Split**: 700 train / 200 val / 100 test (seed=1234, ratio=70%/20%/10%)

## Expected Directory Structure

After downloading and preprocessing, organize the data as follows:

```
data/
├── CVC/
│   ├── train_npz/
│   │   ├── case00000.npz
│   │   ├── case00001.npz
│   │   └── ...
│   ├── val_npz/
│   │   └── ...
│   └── test_npz/
│       └── ...
├── GlaS/
│   ├── train_npz/
│   ├── val_npz/
│   └── test_npz/
└── Kvasir/
    ├── train_npz/
    ├── val_npz/
    └── test_npz/
```

## NPZ File Format

Each `.npz` file should contain:

- `image`: numpy array of shape `(H, W, 3)`, dtype `float32`, range `[0, 1]`
- `label`: numpy array of shape `(H, W)`, dtype `uint8`, values `{0, 1}`

## Preprocessing Logic

### Split Rule

All datasets use the same random seed for reproducibility:

```python
import random
random.seed(1234)
random.shuffle(image_files)
```

The split ratios vary by dataset scale:

| Dataset    | Total | Train Ratio | Val Ratio | Test Ratio |
|------------|-------|-------------|-----------|------------|
| CVC        | 612   | 80%         | 10%       | 10%        |
| GlaS       | 165   | 70%         | 15%       | 15%        |
| Kvasir-SEG | 1000  | 70%         | 20%       | 10%        |

Split counts are computed with `int()` truncation (e.g., 165 * 0.70 = 115).

### Image & Mask Processing

```python
import numpy as np
from PIL import Image

# Load and resize image to (224, 224), convert to RGB
img = Image.open("image.png").convert("RGB")
img = img.resize((224, 224), Image.BILINEAR)
img_array = np.array(img).astype(np.float32) / 255.0  # [0, 1]

# Load and resize mask to (224, 224)
mask = Image.open("mask.png").convert("L")
mask = mask.resize((224, 224), Image.NEAREST)
mask_array = np.array(mask)

# Binarize: all non-zero values become 1 (handles instance-level annotations)
mask_array = (mask_array > 0).astype(np.uint8)

# Save
np.savez_compressed("case00000.npz", image=img_array, label=mask_array)
```
