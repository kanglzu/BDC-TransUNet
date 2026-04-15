"""Medical image dataset loader for binary segmentation tasks."""

import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    """Random rotation and flip."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """Random rotation."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator:
    """Training data augmentation."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # Handle RGB images (H, W, 3)
        if len(image.shape) == 3:
            h, w, c = image.shape
            if h != self.output_size[0] or w != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w, 1), order=3)
                label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        else:
            # Handle grayscale images (H, W)
            x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample


class ValGenerator:
    """Validation generator (resize only, no augmentation)."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if len(image.shape) == 3:
            h, w, c = image.shape
            if h != self.output_size[0] or w != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w, 1), order=3)
                label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        else:
            x, y = image.shape
            if x != self.output_size[0] or y != self.output_size[1]:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample


class MedicalDataset(Dataset):
    """General medical image dataset supporting GLAS, Kvasir, CVC, etc."""
    def __init__(self, base_dir, list_dir, split, transform=None):
        """
        Args:
            base_dir: dataset root directory
            list_dir: directory containing split list files
            split: 'train', 'val', or 'test_vol'
            transform: data augmentation transforms
        """
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        
        list_file = os.path.join(list_dir, f'{split}.txt')
        if os.path.exists(list_file):
            self.sample_list = open(list_file).readlines()
        else:
            # Fallback to directory scan
            print(f"Warning: list file not found: {list_file}, scanning directory instead.")
            self.sample_list = sorted([f.replace('.npz', '') for f in os.listdir(base_dir) if f.endswith('.npz')])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name + '.npz')
        data = np.load(data_path)
        image, label = data['image'], data['label']

        # Ensure RGB 3-channel format
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] == 1:
            image = np.concatenate([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        else:
            if len(image.shape) == 3:
                image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
            else:
                image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}
        
        sample['case_name'] = slice_name
        return sample
