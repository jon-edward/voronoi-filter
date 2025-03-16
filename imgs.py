"""
Generate images for README.md
"""

import pathlib

import numpy as np
from PIL import Image, ImageFilter

from voronoi_filter import voronoi_filter, voronoi_arrs, VoronoiFilterOptions

IMGS_DIR = pathlib.Path(__file__).parent / "imgs"

VORONOI_OPTS = VoronoiFilterOptions(seed=42)


def find_edges_image(image: Image.Image) -> Image.Image:
    image.filter(ImageFilter.FIND_EDGES).convert("L").save(IMGS_DIR / "find_edges.png")


def blurred_edges_image(image: Image.Image) -> Image.Image:
    image.filter(ImageFilter.FIND_EDGES).convert("L").filter(
        ImageFilter.GaussianBlur(radius=2)
    ).save(IMGS_DIR / "blurred_edges.png")


def dithering_images(image: Image.Image) -> Image.Image:
    node_coords, node_colors = voronoi_arrs(image, VORONOI_OPTS)
    out_image_arr = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    for x, y in node_coords:
        out_image_arr[y, x] = 255
    Image.fromarray(out_image_arr).save(IMGS_DIR / "dithering_white.png")

    out_image_arr = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
    for x, y, col in zip(node_coords[:, 0], node_coords[:, 1], node_colors):
        out_image_arr[y, x] = col
    Image.fromarray(out_image_arr).save(IMGS_DIR / "dithering_color.png")


if __name__ == "__main__":
    img = Image.open("input.jpg")

    find_edges_image(img)
    blurred_edges_image(img)
    dithering_images(img)
    voronoi_filter(img, VORONOI_OPTS).save(IMGS_DIR / "voronoi.png")
    voronoi_filter(
        img, VoronoiFilterOptions(distance_metric="sqeuclidean", seed=VORONOI_OPTS.seed)
    ).save(IMGS_DIR / "voronoi_euclidean.png")
