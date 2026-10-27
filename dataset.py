import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile
import cv2
import torch
from scipy import ndimage
from VI_metrics import calculate_vi


def random_rot_flip(image, label, aux_image=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    if aux_image is not None:
        aux_image = np.rot90(aux_image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    if aux_image is not None:
        aux_image = np.flip(aux_image, axis=axis).copy()
    return image, label, aux_image


def random_rotate(image, label, aux_image=None):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)  # order=3 for cubic interpolation
    label = ndimage.rotate(label, angle, order=0, reshape=False)  # order=0 for nearest
    if aux_image is not None:
        aux_image = ndimage.rotate(aux_image, angle, order=3, reshape=False)
    return image, label, aux_image


class RandomAugment(object):
    def __call__(self, data):
        image, label, aux_image = data['image'], data['label'], data['aux_image']

        if random.random() > 0.5:
            image, label, aux_image = random_rot_flip(image, label, aux_image)
        if random.random() > 0.5:
            image, label, aux_image = random_rotate(image, label, aux_image)

        return {'image': image, 'label': label, 'aux_image': aux_image}


class ToTensor(object):

    def __call__(self, data):
        image, label, aux_image = data['image'], data['label'], data['aux_image']
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        aux_image = torch.from_numpy(aux_image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        return {'image': image, 'label': label.long(), 'aux_image': aux_image}


class Resize(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data):
        image, label, aux_image = data['image'], data['label'], data['aux_image']

        # Use cv2.resize for better handling of numpy arrays
        # Use CUBIC interpolation for images and NEAREST for labels
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)
        aux_image = cv2.resize(aux_image, self.size, interpolation=cv2.INTER_CUBIC)

        return {'image': image, 'label': label, 'aux_image': aux_image}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            # Use np.fliplr for horizontal flip
            return {'image': np.fliplr(image).copy(), 'label': np.fliplr(label).copy()}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            # Use np.flipud for vertical flip
            return {'image': np.flipud(image).copy(), 'label': np.flipud(label).copy()}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.2002, 0.2259, 0.1882], std=[0.1255, 0.1096, 0.0813]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # Normalization for numpy arrays
        image = (image - self.mean) / self.std
        return {'image': image, 'label': label}


class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        # Restrict the dataset to GeoTIFF image-label pairs.
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.tif')]
        self.gts = [os.path.join(gt_root, f.replace('_image.tif', '_label.tif')) for f in os.listdir(image_root) if
                    f.endswith('.tif')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.size = size
        # Vegetation indices used by the auxiliary encoder.
        self.supported_vis = ['rvi', 'dvi', 'savi', 'ndwi']

        # Dataset-specific transformation pipeline.
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize(size),
                RandomAugment(),  # Includes rotation and flipping.
                ToTensor()
            ])
        else:  # Validation and testing.
            self.transform = transforms.Compose([
                Resize(size),
                ToTensor()
            ])

    def __getitem__(self, idx):
        image_path = self.images[idx]
        full_image = self.rgb_loader(image_path)
        label = self.binary_loader(self.gts[idx])

        # Normalize image pixel values
        full_image = full_image / 3600.0

        # Prepare main encoder input (NIR, R, G)
        # Assume the original TIF has 4 bands in order: B, G, R, NIR
        # Select NIR, R, G channels (indices 3, 2, 1)
        main_input = full_image[:, :, [3, 2, 1]]

        # Prepare auxiliary encoder input (B + selected VIs)
        # 1. Extract the B band
        blue_channel = full_image[:, :, 0]

        # 2. Compute all supported vegetation indices
        vi_channels = []
        for vi_name in self.supported_vis:
            # calculate_vi reads the spectral bands directly from the source file.
            vi_result = calculate_vi(image_path, vi_type=vi_name)
            if vi_result is not None:
                vi_channels.append(vi_result)
        
        # 3. Stack B band and all VI channels into a multi-channel auxiliary image
        #    First add B band to the list
        all_aux_channels = [blue_channel] + vi_channels
        #    Use np.stack along the last dimension
        aux_input = np.stack(all_aux_channels, axis=-1)

        # Map label values from [1, 21] to [0, 20]
        label = label - 1

        data = {'image': main_input, 'label': label, 'aux_image': aux_input}
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        # Read .tif images using tifffile
        return tifffile.imread(path)

    def binary_loader(self, path):
        # Read .tif labels using tifffile, ensuring correct dtype and values
        # For single-band label TIFs, it returns a 2D numpy array directly
        return tifffile.imread(path)


class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.tif')]
        self.gts = [gt_root + f.replace('_image.tif', '_label.tif') for f in os.listdir(image_root) if
                    f.endswith('.tif')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            Resize(size),
            Normalize(),
            ToTensor()
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = tifffile.imread(self.images[self.index])

        # Apply the same channel order used during training.
        # Assume the original TIF has 4 bands in order: B, G, R, NIR
        if image.shape[2] == 4:
            # Select NIR, R, G channels (indices 3, 2, 1)
            image = image[:, :, [3, 2, 1]]

        image = image / 3600.0

        gt = tifffile.imread(self.gts[self.index])
        gt = gt if len(gt.shape) == 2 else gt[:, :, 0]

        # Apply -1 to test labels to ensure consistent classes during evaluation
        gt_for_eval = gt - 1

        data = {'image': image, 'label': gt_for_eval}
        data = self.transform(data)

        image_tensor = data['image'].unsqueeze(0)

        name = self.images[self.index].split(os.sep)[-1]

        self.index += 1
        return image_tensor, gt_for_eval, name

    def rgb_loader(self, path):
        return tifffile.imread(path)

    def binary_loader(self, path):
        # Read the single-band label raster.
        return tifffile.imread(path)
