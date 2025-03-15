"""
Implementation of Voronoi image filtering

See README.md for details
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter
from scipy.spatial.distance import cdist
from tqdm import tqdm


@dataclass(frozen=True)
class VoronoiFilterOptions:
    seed: int | None = None
    """
    The seed to use for the random number generator
    """

    blur_radius: float = 5.0
    """
    The radius of the Gaussian blur applied to the image post-"find edges"
    """

    baseline_probability: float = 0.005
    """
    The baseline probability for all pixels in the image to be selected as voronoi nodes
    """

    probability_power: float = 3.0
    """
    The exponent applied to the probabilities of voronoi nodes
    """

    proportion_points: float = 0.15
    """
    The proportion of voronoi nodes to sample from the image
    """

    distance_metric: str = "cityblock"
    """
    The distance metric to use when finding the closest voronoi node

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html 
    """


def voronoi_arrs(image: Image.Image, opts: VoronoiFilterOptions = VoronoiFilterOptions()) -> np.ndarray:
    """
    Returns an array of coordinates and colors for voronoi nodes
    """
    image_edges = (
        image.filter(ImageFilter.FIND_EDGES)
        .convert("L")
        .filter(ImageFilter.GaussianBlur(radius=opts.blur_radius))
    )
    probabilities = (
        ((np.array(image_edges).astype(np.float32)) / 255.0) + opts.baseline_probability
    ) ** opts.probability_power
    probabilities = (probabilities / np.sum(probabilities)).swapaxes(0, 1).flatten()

    rng = np.random.default_rng(opts.seed)
    indices = rng.choice(
        len(probabilities),
        int(len(probabilities) * opts.proportion_points),
        p=probabilities,
        replace=False,
    )

    references = np.column_stack((indices % image.size[1], indices // image.size[1]))
    colors = np.array([image.getpixel((y, x)) for x, y in references])

    return references, colors


def _coords(row, col) -> np.ndarray:
    """
    Returns an array of coordinates

    For example, coords(2, 3) returns:
    [[[0 0]
      [0 1]
      [0 2]]

     [[1 0]
      [1 1]
      [1 2]]]
    """
    return np.array(list(np.ndindex((row, col)))).reshape(row, col, 2)


def _dist_argmin(a: np.ndarray, references: np.ndarray, metric: str) -> np.ndarray:
    """
    Returns the index of the reference that is closest to a given coordinate in a
    """
    return np.argmin(cdist(a, references, metric=metric), axis=1)


def voronoi_filter(
    image: Image.Image, opts: VoronoiFilterOptions = VoronoiFilterOptions()
) -> Image.Image:
    """
    Returns a voronoi filtered image
    """
    node_coords, node_colors = voronoi_arrs(image, opts)
    
    # Generate the array of closest voronoi nodes per pixel
    arr = _coords(int(image.size[1]), int(image.size[0]))
    out_idx = np.array(
        [_dist_argmin(a, node_coords, opts.distance_metric) for a in tqdm(arr)]
    )

    # Convert the array of closest voronoi nodes per pixel to an image
    out = []
    for out_arr in out_idx:
        out.append([node_colors[i] for i in out_arr])
    out = np.array(out)
    return Image.fromarray(out.astype(np.uint8), mode=image.mode)
