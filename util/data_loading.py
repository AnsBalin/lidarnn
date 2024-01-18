from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import os
import json
from functools import lru_cache

from torchvision.transforms.functional import pil_to_tensor


def create_data_index(data_path, index_file=None, include_metadata=False):
    """Scan data_path to find all features and masks.

    Args:
        data_path: path to preprocessed lidar data. Contents would be subdirectories like 
            'LIDAR-DTM-1m-2022-SS21nw'
        index_file: output json filename. If None will not write to any file.

    Returns: 
        dictionary containing N number of os_tiles found, and data_index which indexes each tile 
        directory containing a features.png and mask.png

    """

    # The preprocessed data directories eg ['LIDAR-DTM-1m-2022-NZ09se']
    os_squares = [f.name for f in os.scandir(data_path) if f.is_dir()]

    data_index = {}
    i = 0

    for os_square in os_squares:
        tiles = [subd.name for subd in os.scandir(
            os.path.join(data_path, os_square))]

        for tile in tiles:
            tile_path = os.path.join(data_path, os_square, tile)
            files = [f.name for f in os.scandir(tile_path)]

            if 'features.png' in files and 'mask.png' in files:

                data_index[i] = {'path': tile_path}

                if include_metadata and 'metadata.json' in files:
                    with open(os.path.join(tile_path, 'metadata.json'), 'r') as json_file:
                        metadata = json.load(json_file)
                        data_index[i]['metadata'] = metadata

                i = i + 1

    if index_file:
        with open(index_file, 'w') as fp:
            json.dump(data_index, fp, default=lambda _: '<n/a>')

    return data_index


@lru_cache
def get_data_index(data_path, index_file=None):

    if index_file is None or index_file not in os.listdir():
        return create_data_index(data_path)

    with open(index_file, 'r') as json_file:
        data_index = json.load(json_file)

    data_index = {int(key): value for key, value in data_index.items()}
    return data_index


def preprocess(pil_img, is_mask=False):
    w, h = pil_img.size

    img = np.asarray(pil_img)

    if is_mask:
        mask = np.zeros((w, h), dtype=np.int64)

        mask[img > 254.] = 1

        return mask

    else:
        img = img.transpose((2, 0, 1))
        img = img / 255.0

        return img


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

        feature = preprocess(feature, False)
        mask = preprocess(mask, True)

        return {
            'feature': torch.as_tensor(feature.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }

    def preprocess(self, pil_img, is_mask=False):
        w, h = pil_img.size

        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((w, h), dtype=np.int64)

            mask[img > 254.] = 1

            return mask

        else:
            img = img.transpose((2, 0, 1))
            img = img / 255.0

            return img


class LidarDataset(Dataset):
    def __init__(self, data_path, index_file='data_index.json'):
        self.data_index = get_data_index(data_path, index_file)

    def __len__(self):
        return len(self.data_index)

    @lru_cache(512)
    def __getitem__(self, index):

        path = self.data_index[index]['path']
        print(path)

        feature = Image.open(f"{path}/features.png")
        mask = Image.open(f"{path}/mask.png")

        print('got feature and mask')
        feature = preprocess(feature, False)
        mask = preprocess(mask, True)
        print('processed feature and mask')

        return {
            'feature': torch.as_tensor(feature.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }


if __name__ == '__main__':

    dataset = LidarDataset('/mnt/d/lidarnn')
    f = dataset[0]['feature']
    m = dataset[0]['mask']
