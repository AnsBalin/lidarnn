from torch.utils.data import Dataset
from PIL import Image
import torch


class LidarDatasetSynthetic(Dataset):
    def __init__(self, feature_dir, mask_dir, N=100):
        self.feature_dir = feature_dir
        self.mask_dir = mask_dir

        self.features = []
        self.masks = []

        for i in range(N):
            image = Image.open(
                "{path:s}/feature_{i:d}.png".format(path=feature_dir, i=i))
            # image_pixels = image.load()

            mask = Image.open(
                "{path:s}/mask_{i:d}.png".format(path=feature_dir, i=i))
            # mask_pixels = mask.load()

            self.features.append(image)
            self.masks.append(mask)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        mask = self.masks[index]

        return {
            'feature': torch.as_tensor(feature.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }
