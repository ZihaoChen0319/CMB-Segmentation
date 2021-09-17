import numpy as np
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.measurements import label
from scipy import ndimage
from scipy.ndimage import filters


def find_binary_object(array, connectivity=1):
    array = np.atleast_1d(array.astype(np.bool))
    footprint = generate_binary_structure(array.ndim, connectivity)
    labelmap, n_obj = label(array, footprint)
    return labelmap, n_obj


def find_local_maxima(img, neighbor_size=5, threshold=0.1):
    img_max = filters.maximum_filter(img, neighbor_size)
    maxima = (img == img_max)
    img_min = filters.minimum_filter(img, neighbor_size)
    diff = (img_max - img_min) > threshold
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    maxima_list = ndimage.center_of_mass(img, labeled, range(1, num_objects + 1))
    return labeled, num_objects, maxima_list


if __name__ == '__main__':
    pass
