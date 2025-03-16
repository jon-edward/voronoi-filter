"""
Run voronoi filter on an image
"""

from PIL import Image

from voronoi_filter import voronoi_filter, VoronoiFilterOptions

if __name__ == "__main__":
    img = Image.open("input.jpg")
    out = voronoi_filter(img, VoronoiFilterOptions())
    out.save("out.png")
    out.show()
