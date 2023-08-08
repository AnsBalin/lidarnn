import shapefile
import matplotlib.pyplot as plt
import random
from io import BytesIO
from PIL import Image
import numpy as np


def make_mask(shape, L):
    points = shape.points
    (xmin, ymin, xmax, ymax) = shape.bbox
    xs = np.array([i[0] for i in points])
    ys = np.array([i[1] for i in points])
    center = ((xmin+xmax)/2.0, (ymin+ymax)/2.0)
    L = 500
    x_off = np.random.uniform(0, L, 1)
    y_off = np.random.uniform(0, L, 1)

    xs_ = xs - center[0] + x_off
    ys_ = ys - center[1] + y_off

    return xs_, ys_


def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image.

    Parameters:
        image (numpy.array): Input image.
        mean (float): Mean of the Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.array: Image with added Gaussian noise.
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def make_image(shape, L, type, size):
    xs, ys = make_mask(shape, L)

    if type == 'mask':
        plt.fill(xs, ys)
    else:
        plt.plot(xs, ys)
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.gcf().canvas.draw()
    bbox = plt.gca().get_window_extent().transformed(
        plt.gcf().dpi_scale_trans.inverted())

    # Hide axes
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)

    # Save to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches=bbox,
                pad_inches=0, dpi=size/bbox.width)
    buf.seek(0)
    plt.close()

    # Use PIL to convert to an array
    img = Image.open(buf)
    img_array = np.array(img)

    if type == 'image':
        img_array = add_gaussian_noise(img_array, 20)

    return img_array


if __name__ == '__main__':
    sf = shapefile.Reader("monuments/Scheduled_Monuments.shp")
    shapes = sf.shapes()

    for i in range(100):
        shape = shapes[i]
        seed = len(shape.points)
        L = 500
        np.random.seed(seed)
        image = make_image(shape, L, 'image', 128)
        np.random.seed(seed)
        mask = make_image(shape, L, 'mask', 128)

        im = Image.fromarray(image)
        im.save("synthetic/image_{i:d}.png".format(i=i))

        mk = Image.fromarray(mask)
        mk.save("synthetic/mask_{i:d}.png".format(i=i))
