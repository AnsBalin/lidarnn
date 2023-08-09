
from affine import Affine
import geopandas as gpd
import numpy as np
from PIL import Image
from tqdm import tqdm

from rasterio.features import rasterize
from rasterio.features import geometry_mask


def make_mask(geom, L):
    (xmin, ymin, xmax, ymax) = geom.bounds
    center = ((xmin+xmax)/2.0, (ymin+ymax)/2.0)

    x_off = np.random.uniform(0, L, 1)
    y_off = np.random.uniform(0, L, 1)

    s = 2
    transform = Affine(s, 0, (center[0])-s*x_off,
                       0, s, (center[1])-s*y_off)

    mask = geometry_mask([geom],
                         transform=transform, invert=True, out_shape=(L, L))

    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_image, transform


def add_noise(image, mu, sigma):
    row, col = image.shape
    image2 = (1-np.dstack([image, image, image])) * 255
    gauss = np.random.normal(mu, sigma, (row, col, 3))
    gauss = gauss.reshape(row, col, 3)
    noisy = image2 + gauss
    feature = np.clip(noisy, 0, 255).astype(np.uint8)

    return feature


def make_feature(geom, L, transform):
    boundary = geom.boundary
    image = np.zeros((L, L), dtype=np.uint8)

    rasterize([boundary], out=image, transform=transform)

    feature = add_noise(image, 0, 100)
    feature_image = Image.fromarray(feature)
    return feature_image


if __name__ == '__main__':

    sf = gpd.read_file("monuments/Scheduled_Monuments.shp")

    for i in tqdm(range(1000)):
        geom = sf['geometry'][i]
        L = 256
        mask_image, transform = make_mask(geom, L)
        feature_image = make_feature(geom, L, transform)

        mask_image.save("synthetic/mask_{i:d}.png".format(i=i))
        feature_image.save("synthetic/feature_{i:d}.png".format(i=i))
