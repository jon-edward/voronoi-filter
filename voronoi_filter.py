"""
Implementation of Voronoi image filtering

See README.md for details
"""

from dataclasses import dataclass
import itertools

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


def voronoi_arrs(
    image: Image.Image, opts: VoronoiFilterOptions = VoronoiFilterOptions()
) -> np.ndarray:
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
    probabilities = (probabilities / np.sum(probabilities)).flatten()

    rng = np.random.default_rng(opts.seed)
    indices = rng.choice(
        len(probabilities),
        int(len(probabilities) * opts.proportion_points),
        p=probabilities,
        replace=False,
    )

    references = np.column_stack((indices % image.size[0], indices // image.size[0]))
    colors = np.array([image.getpixel((x, y)) for x, y in references])

    return references, colors


def _coords(row, col) -> np.ndarray:
    """
    Returns an array of coordinates

    For example, _coords(2, 3) returns:
    [[0 0]
     [0 1]
     [0 2]
     [1 0]
     [1 1]
     [1 2]]
    """

    return np.array(list(itertools.product(range(row), range(col))))


def _dist_argmin(a: np.ndarray, references: np.ndarray, metric: str) -> np.ndarray:
    """
    Returns the indices of the references that are closest to each coordinate in a
    """
    dists = cdist(a, references, metric=metric)
    a = np.argmin(dists, axis=1)
    return a


def voronoi_filter(
    image: Image.Image, opts: VoronoiFilterOptions = VoronoiFilterOptions()
) -> Image.Image:
    """
    Returns a voronoi filtered image
    """
    node_coords, node_colors = voronoi_arrs(image, opts)

    # Generate the array of closest voronoi nodes per pixel
    # This is done in chunks to avoid running out of memory
    out_idx = np.concatenate(
        [
            _dist_argmin(a, node_coords, opts.distance_metric)
            for a in tqdm(np.array_split(_coords(*image.size), len(node_coords)))
        ]
    )

    # Convert the array of closest voronoi nodes per pixel to an image
    out = (
        np.array([node_colors[idx] for idx in out_idx])
        .reshape((image.size[0], image.size[1], -1))
        .transpose((1, 0, 2))
    )
    if out.shape[2] == 1:
        out = out[:, :, 0]

    return Image.fromarray(out.astype(np.uint8), mode=image.mode)
