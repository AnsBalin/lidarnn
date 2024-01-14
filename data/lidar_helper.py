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


import cProfile

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
def _get_shape(shapes_directory, id):
    """Wrapper for accessing shapes from config paths"""
    if id == 'gb':
        path = os.path.join(shapes_directory, GB_SHAPEFILE_PATH)
    elif id == 'monuments':
        path = os.path.join(shapes_directory, MONUMENTS_SHAPEFILE_PATH)

    return gpd.read_file(path)


def get_gb(shapes_directory):
    """Returns a gpd table with one element containing the geometry of the GB coastline"""
    return _get_shape(shapes_directory, 'gb')


def get_monuments(shapes_directory):
    """
    Returns a gpd table for the Historic England Scheduled Monuments dataset. 
    Each row contains geometry + metadata of a single monument.
    """
    return _get_shape(shapes_directory, 'monuments')


def files_to_unzip(folder_path):
    """Scans folder path and returns list of DTM directories (unzipped) and DTM .zip files"""

    cwd = os.getcwd()

    try:
        os.chdir(folder_path)
        files = os.listdir(folder_path)

        zip_files = [f for f in files
                     if f.startswith(DTM_PREFIX) and f.endswith('.zip')]

        # remove .zip extension
        zip_files = [z.rsplit('.', 1)[0] for z in zip_files]

        dirs = [f.name for f in os.scandir()
                if f.name.startswith(DTM_PREFIX) and f.is_dir()]

    finally:
        os.chdir(cwd)

    return dirs, zip_files


def unzip_files_in_directory(folder_path=None, zip_files=None, logger=logging, delete_zip=False):
    """
    Unzips all zip files in path matching DTM_PREFIX. 
    Only extracts files if target path does not exist.
    """

    if zip_files is None:
        _, zip_files = files_to_unzip(folder_path)

    cwd = os.getcwd()
    try:
        os.chdir(folder_path)

        for zip_file in zip_files:
            # we now remove zip extension earlier
            # dir_name = zip_file.rsplit('.', 1)[0]  # Remove the .zip extension
            dir_name = zip_file

            if not os.path.exists(dir_name):
                logger.info(f"Unzipping {zip_file} into {dir_name}")
                os.makedirs(dir_name)

                with zipfile.ZipFile(zip_file + '.zip', 'r') as zip_ref:
                    zip_ref.extractall(dir_name)
                logger.info(f"Unzipped {zip_file}")

                if delete_zip:
                    os.remove(zip_file + '.zip')

            else:
                logger.info(f"{dir_name} already exists. Skipping...")

    except Exception as e:
        logger.error(f"{zip_file}: {e}")
        if os.path.exists(dir_name):
            os.removedirs(dir_name)

    finally:
        os.chdir(cwd)


def hillshade(array, azimuth=315, angle_altitude=45):
    """
    Creates a hillshade of a given numpy array. This is the same as the hillshade 
    from earthpy 0.9.4.

    Original documentation:
    https://earthpy.readthedocs.io/en/latest/gallery_vignettes/plot_dem_hillshade.html?highlight=hillshade
    """
    # Some -1e38 values exist, this floors it to stop runtime overflow warning
    array[array < -1e6] = -1e6

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


def files_to_process(data_raw_path, data_out_path, output_image_size):

    n_subtiles = _n_subtiles(DTM_SIZE, output_image_size) ** 2

    done = []

    # The unzipped data directories eg ['LIDAR-DTM-1m-2022-NZ09se']
    full_todo = [f.name for f in os.scandir(data_raw_path) if f.is_dir()]

    # The preprocessed data directories eg ['LIDAR-DTM-1m-2022-NZ09se']
    png_dirs = [f.name for f in os.scandir(data_out_path) if f.is_dir()]

    # Check the contents of these directories is sensible. They should have
    # n_subtiles subdirectories
    for png_dir in png_dirs:
        subdirectories = [subd.name for subd in os.scandir(
            os.path.join(data_out_path, png_dir))]
        if len(subdirectories) == n_subtiles:
            done.append(png_dir)

    todo = [d for d in full_todo if d not in done]

    return done, todo


def create_features_and_masks(tile_ref, data_raw_path, shapes_directory='./', L=256, channels=3, logger=logging.getLogger()):
    elevation, transform, tile_geometry = get_tile(tile_ref, data_raw_path)
    logger.debug(f"DONE: get_tile() for {tile_ref}")

    gb = get_gb(shapes_directory)
    monuments = get_monuments(shapes_directory)
    logger.debug(f"DONE: get_gb() and get_monuments() for {tile_ref}")

    hillstack = hill_stack(elevation, channels, 10)
    logger.debug(f"DONE: hill_stack() for {tile_ref}")

    overlaps = monuments.intersects(tile_geometry['geometry'].any())
    monuments_geoms = list(monuments[overlaps]['geometry'])
    gb_geoms = gb.intersection(tile_geometry)

    if len(monuments_geoms):
        monuments_mask = geometry_mask(
            monuments_geoms, transform=transform, invert=False, out_shape=elevation.shape)
    else:
        monuments_mask = np.ones(elevation.shape)

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

        # count the fraction of tile with missing lidar data (gradiens in x and y precisely 0)
        lidar_coverage = ((np.diff(hillstack_mini, axis=0, prepend=hillstack_mini[(0,), :]) +
                           np.diff(hillstack_mini, axis=1, prepend=hillstack_mini[:, (0,)])) != 0).sum() \
            / np.size(hillstack_mini)

        # count the fraction of tile overlapping with a monument
        monument_coverage = (monuments_mask_mini == 0).sum() / \
            np.size(monuments_mask_mini)

        # Count the fraction of pixels representing land.
        land_coverage = np.count_nonzero(
            boundary_mask_mini) / np.size(boundary_mask)

        data.append({
                    'hillstack': hillstack_mini,
                    'monuments': monuments_mask_mini,
                    'metadata': {
                        'id': f"{tile_ref}_{x0}_{y0}",
                        'tile_ref': tile_ref,
                        'index_x': x0.astype('int'),
                        'index_y': y0.astype('int'),
                        'origin_easting': easting,
                        'origin_northing': northing,
                        'land_coverage': land_coverage,
                        'lidar_coverage': lidar_coverage,
                        'monument_coverage': monument_coverage,
                        'subtile_overlaps': subtile_overlaps

                    }})

    return data


def process_dtm(dtm_dirs, data_raw_path, data_out_path, shapes_directory, logger, output_image_size=256):
    """Perform LIDAR image processing steps on a list of unzipped directories.

    Args:
        dtm_dirs: A list of directories, for example ['LIDAR-DTM-1m-2022-NT60se']
        logger: logger object implementing .info(), .warning() etc.

    Returns:
        None
    """

    for dtm_dir in dtm_dirs:

        # dtm_dir for instance './LIDAR-DTM-1m-2022-SE32ne'
        # Tile Ref is SE32ne
        tile_ref = dtm_dir.split('-')[-1]

        logger.info(f"Creating features and masks for {tile_ref}")
        subtiles = create_features_and_masks(
            tile_ref, data_raw_path, shapes_directory, L=output_image_size, channels=3, logger=logger)
        logger.info(f"Created subtiles for {tile_ref}")
        for subtile in subtiles:
            features = subtile['hillstack']
            mask = subtile['monuments']
            metadata = subtile['metadata']
            id_str = metadata['id']
            path = os.path.join(data_out_path, dtm_dir, id_str)
            if not os.path.exists(path):
                os.makedirs(path)

            img_features = Image.fromarray(features.astype(np.uint8))
            img_mask = Image.fromarray(mask.astype(np.uint8))

            img_features.save(f'{path}/features.png')
            img_mask.save(f'{path}/mask.png')

            with open(f'{path}/metadata.json', 'w') as fp:
                json.dump(metadata, fp, default=lambda _: '<n/a>')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    cProfile.run(
        "process_dtm(['LIDAR-DTM-1m-2022-NZ09se','LIDAR-DTM-1m-2022-NZ09nw','LIDAR-DTM-1m-2022-NT60ne'],'/mnt/d/lidarnn_raw_new', '/mnt/d/lidarnn', '/mnt/d/lidarnn_shapes', logger, 256)")
