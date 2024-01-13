import logging
import zipfile
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import geometry_mask
from functools import lru_cache
import logging
from itertools import product
from PIL import Image
import json
from shapely import Polygon
from tqdm import tqdm

# Globals - Move to config later
DTM_PREFIX = 'LIDAR-DTM-1m-2022'
DTM_SIZE = 5000

# Absolute paths
DTM_RAW_PATH = '/mnt/d/lidarnn_raw'
DATA_PATH = '/mnt/d/lidarnn'

# Relative paths (relative to lidarnn/data/)
GB_SHAPEFILE_PATH = 'gb/infuse_gb_2011.shp'
MONUMENTS_SHAPEFILE_PATH = 'monuments/Scheduled_Monuments.shp'
# Relative paths (relative to DATA_PATH/{tile_ref} )
FEATURES_PATH = 'features'
MASKS_PATH = 'masks'
META_PATH = 'meta'


def _n_subtiles(M, L):
    return (int(np.ceil(M / L)))


@lru_cache(maxsize=None)
def _get_shape(id):
    """Wrapper for accessing shapes from config paths"""
    if id == 'gb':
        path = GB_SHAPEFILE_PATH
    elif id == 'monuments':
        path = MONUMENTS_SHAPEFILE_PATH

    return gpd.read_file(path)


def get_gb():
    """Returns a gpd table with one element containing the geometry of the GB coastline"""
    return _get_shape('gb')


def get_monuments():
    """
    Returns a gpd table for the Historic England Scheduled Monuments dataset. 
    Each row contains geometry + metadata of a single monument.
    """
    return _get_shape('monuments')


def unzip_files_in_directory(folder_path=None, zip_files=None, logger=logging):
    """
    Unzips all zip files in path matching DTM_PREFIX. 
    Only extracts files if target path does not exist.
    """

    cwd = os.getcwd()

    try:
        os.chdir(folder_path)

        if zip_files is None:
            files = os.listdir(folder_path)

            zip_files = [f for f in files
                         if f.startswith(DTM_PREFIX) and f.endswith('.zip')]

        for zip_file in zip_files:
            dir_name = zip_file.rsplit('.', 1)[0]  # Remove the .zip extension

            if not os.path.exists(dir_name):
                logger.info(f"Unzipping {zip_file} into {dir_name}")
                os.makedirs(dir_name)

                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(dir_name)
                logger.info(f"Unzipped {zip_file}")
            else:
                logger.info(f"{dir_name} already exists. Skipping...")

    finally:
        os.chdir(cwd)


def hillshade(array, azimuth=315, angle_altitude=45):
    """
    Creates a hillshade of a given numpy array. This is the same as the hillshade 
    from earthpy 0.9.4.

    Original documentation:
    https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_dem_hillshade.html?highlight=hillshade
    """
    azimuth = 360.0 - azimuth
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = np.radians(azimuth)
    altituderad = np.radians(angle_altitude)

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * \
        np.cos(slope) * np.cos((azimuthrad - np.pi / 2.0) - aspect)

    return 255 * (shaded + 1) / 2.


def get_tile(tile_ref, path=DTM_RAW_PATH):
    """Reads DTM raster from disk and returns elevation map and affine transform."""

    file = f'{path}/{DTM_PREFIX}-{tile_ref}/{tile_ref}_DTM_1m.tif'
    shapefile = f'{path}/{DTM_PREFIX}-{tile_ref}/index/{tile_ref}_DTM_1m.shp'

    with rasterio.open(file) as src:
        transform = src.transform
        elevation = src.read(1)

    geometry = gpd.read_file(shapefile)

    return elevation, transform, geometry


def hill_stack(elevation, channels=1, altitude=10):
    """
    Returns an n-channel array representing a stack of hillshades. 

    Output shape will be (elevation.shape[0], elevation.shape[1], channels)

    Parameters: 
        elevation - DTM elevation map (2-d numpy array)
        channels - number of channels in the final stack.

    Channel n=0 will always have azimuth 0.
    If channels == 2, channel n=1 will have azimuth pi/2.
    If channels > 2, then the nth hillshade will have an azimuth 2*pi*n/channels.
    """

    if channels < 1:
        raise ValueError('Number of channels in hillshade stack must be > 0.')

    azimuths = np.linspace(0., 360., channels + 1)
    if channels == 2:
        azimuths[1] = 90.0

    hill_stack = [hillshade(elevation, azimuth, altitude)
                  for azimuth in azimuths[:-1]]

    return np.stack(hill_stack, axis=2)


def create_features_and_masks(tile_ref, raw_data_path, L=256, channels=3):
    elevation, transform, tile_geometry = get_tile(tile_ref, raw_data_path)
    gb = get_gb()
    monuments = get_monuments()
    hillstack = hill_stack(elevation, channels, 10)

    overlaps = monuments.intersects(tile_geometry['geometry'].any())
    monuments_geoms = list(monuments[overlaps]['geometry'])
    gb_geoms = gb.intersection(tile_geometry)

    if len(monuments_geoms):
        monuments_mask = geometry_mask(
            monuments_geoms, transform=transform, invert=False, out_shape=elevation.shape)
    else:
        monuments_mask = np.ones((L, L))

    boundary_mask = geometry_mask(
        gb_geoms, transform=transform, invert=True, out_shape=elevation.shape)

    monuments_mask = 255. * monuments_mask
    boundary_mask = 255. * boundary_mask

    # Split elevation map into smaller images. M = 5000 for the 5km*5km 1m DTM data
    assert elevation.shape[0] == DTM_SIZE
    M = DTM_SIZE

    # 20x20 images for M = 5000 and L = 256 images will overlap by 6 pixels each
    n_subtiles = _n_subtiles(M, L)
    origins = np.linspace(0, M * (1. - 1. / n_subtiles),
                          n_subtiles).astype('int')

    tile_min_easting = tile_geometry.bounds.iloc[0].minx
    tile_max_northing = tile_geometry.bounds.iloc[0].maxy

    data = []

    for x0, y0 in product(origins, origins):
        # create LxL tile
        hillstack_mini = hillstack[x0:x0+L, y0:y0+L, :]
        monuments_mask_mini = monuments_mask[x0:x0 + L, y0:y0+L]
        boundary_mask_mini = boundary_mask[x0:x0 + L, y0:y0+L]

        # northing is flipped because [0,0] element refers to top-left of image
        easting = tile_min_easting + x0
        northing = tile_max_northing - y0 - L

        subtile_coords = ((easting, northing),
                          (easting, northing + L),
                          (easting + L, northing + L),
                          (easting + L, northing),
                          (easting, northing)
                          )

        subtile_shape = Polygon(subtile_coords)

        subtile_overlaps = monuments[monuments.intersects(subtile_shape)]
        subtile_overlaps = subtile_overlaps.to_dict('records')

        # Count the fraction of pixels representing land.
        land_coverage = np.count_nonzero(
            boundary_mask_mini) / np.size(boundary_mask)

        data.append({
                    'hillstack': hillstack_mini,
                    'monuments': monuments_mask_mini,
                    'metadata': {
                        'id': f'{tile_ref}_{x0}_{y0}',
                        'tile_ref': tile_ref,
                        'index': (x0, y0),
                        'origin': (easting, northing),
                        'land_coverage': land_coverage,
                        'subtile_overlaps': subtile_overlaps

                    }})

    return data


def process_dtm(dtm_dirs, raw_data_path, out_data_path, logger, output_image_size=256):
    """Perform LIDAR image processing steps on a list of unzipped directories.

    Args:
        dtm_dirs: A list of directories, for example ['LIDAR-DTM-1m-2022-NT60se']
        logger: logger object implementing .info(), .warning() etc.

    Returns:
        None
    """

    cwd = os.getcwd()
    try:
        os.chdir(raw_data_path)
        dtm_dirs = [f.path for f in os.scandir() if f.is_dir()
                    and any([d in f.path for d in dtm_dirs])]
    finally:
        os.chdir(cwd)

    data_dirs = os.listdir(out_data_path)
    for dtm_dir in dtm_dirs:

        n_subtiles = _n_subtiles(DTM_SIZE, output_image_size)

        # dtm_dir for instance './LIDAR-DTM-1m-2022-SE32ne'
        # Tile Ref is SE32ne
        tile_ref = dtm_dir.split('-')[-1]

        existing_dirs = [f for f in data_dirs if f.startswith(tile_ref)]

        expected = set(['features.npy', 'features.png',
                        'mask.npy', 'mask.png', 'metadata.json'])

        def unexpected_files(dir):
            dir_files = set(os.listdir(os.path.join(raw_data_path, dir)))
            return dir_files != expected

        subtiles_not_done = list(map(unexpected_files, existing_dirs))

        if len(existing_dirs) < n_subtiles or any(subtiles_not_done):
            logger.info(f"Creating features and masks for {tile_ref}")
            subtiles = create_features_and_masks(
                tile_ref, raw_data_path, L=output_image_size, channels=3)

            for subtile in subtiles:
                features = subtile['hillstack']
                mask = subtile['monuments']
                metadata = subtile['metadata']
                id_str = metadata['id']
                path = os.path.join(out_data_path, id_str)
                if not os.path.exists(path):
                    os.makedirs(path)
                # np.save(f'{path}/features.npy', features)
                # np.save(f'{path}/mask.npy', mask)

                img_features = Image.fromarray(features.astype(np.uint8))
                img_mask = Image.fromarray(mask.astype(np.uint8))

                img_features.save(f'{path}/features.png')
                img_mask.save(f'{path}/mask.png')

                with open(f'{path}/metadata.json', 'w') as fp:
                    json.dump(metadata, fp, default=lambda _: '<n/a>')


if __name__ == '__main__':

    # logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    # Look for zip files and extract if not already extracted
    print(f'Unzipping files in directory {DTM_RAW_PATH}')
    unzip_files_in_directory()

    dtm_dirs = [f.path for f in os.scandir() if f.is_dir()]
    data_dirs = os.listdir(DATA_PATH)

    for dtm_dir in (pbar := tqdm(dtm_dirs)):

        # output image size
        L = 256
        n_subtiles = _n_subtiles(DTM_SIZE, L)

        # dtm_dir is like './LIDAR-DTM-1m-2022-SW32ne'
        tile_ref = dtm_dir.split('-')[-1]

        existing_dirs = [f for f in data_dirs if f.startswith(tile_ref)]

        expected = set(['features.npy', 'features.png',
                       'mask.npy', 'mask.png', 'metadata.json'])

        def unexpected_files(dir):
            dir_files = set(os.listdir(os.path.join(DATA_PATH, dir)))
            return dir_files != expected

        subtiles_not_done = list(map(unexpected_files, existing_dirs))

        if len(existing_dirs) < n_subtiles or any(subtiles_not_done):
            pbar.set_description(f'Creating features and masks for {tile_ref}')

            subtiles = create_features_and_masks(tile_ref, L=L, channels=3)

            for subtile in subtiles:

                features = subtile['hillstack']
                mask = subtile['monuments']
                metadata = subtile['metadata']
                id_str = metadata['id']
                pbar.set_description(
                    f'Processing subtile {id_str}')
                path = os.path.join(DATA_PATH, id_str)
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(f'{path}/features.npy', features)
                np.save(f'{path}/mask.npy', mask)

                img_features = Image.fromarray(features.astype(np.uint8))
                img_mask = Image.fromarray(mask.astype(np.uint8))

                img_features.save(f'{path}/features.png')
                img_mask.save(f'{path}/mask.png')

                with open(f'{path}/metadata.json', 'w') as fp:
                    json.dump(metadata, fp, default=lambda o: '<n/a>')
        else:

            pbar.set_description(f'Done {tile_ref}')
