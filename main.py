"""
Run voronoi filter on an image
"""

from PIL import Image

from voronoi_filter import voronoi_filter

OUT_SCALE = 1.0

if __name__ == "__main__":
    img = Image.open("input.jpg")
    img = img.resize((int(img.size[0] * OUT_SCALE), int(img.size[1] * OUT_SCALE)))
    out = voronoi_filter(img)

    out.save("out.png")
    out.show()
