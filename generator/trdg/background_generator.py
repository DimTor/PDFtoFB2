import cv2
import math
import os
import random as rnd
import numpy as np

from PIL import Image, ImageDraw, ImageFilter


def gaussian_noise(height: int, width: int) -> Image:
    """
    Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add gaussian noise
    cv2.randn(image, 235, 10)

    return Image.fromarray(image).convert("RGBA")


def color(height: int, width: int) -> Image:
    """
    Create a plain white background
    """

    return Image.new("RGB", (width, height), (tuple(np.random.randint(150, 255, size=3)))).convert("RGBA")

def color_with_noise(height: int, width: int) -> Image:
    """
    Create a plain white background
    """
    img_ar = np.array(Image.new("RGB", (width, height), (tuple(np.random.randint(150, 255, size=3)))))
    img_ar_np = np.asarray(img_ar)
    average_color_row = np.average(img_ar_np, axis=0)
    average_color = np.average(average_color_row, axis=0)
    cv2.randn(img_ar, tuple([int(i) for i in average_color]), (10, 10, 10))
    return Image.fromarray(img_ar).convert("RGBA")


def plain_white(height: int, width: int) -> Image:
    """
    Create a plain white background
    """

    return Image.new("L", (width, height), 255).convert("RGBA")


def quasicrystal(height: int, width: int) -> Image:
    """
    Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """

    image = Image.new("L", (width, height))
    pixels = image.load()

    frequency = rnd.random() * 30 + 20  # frequency
    phase = rnd.random() * 2 * math.pi  # phase
    rotation_count = rnd.randint(10, 20)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            pixels[kw, kh] = c  # grayscale
    return image.convert("RGBA")


def image(height: int, width: int, image_dir: str) -> Image:
    """
    Create a background with a image
    """
    images = os.listdir(image_dir)

    if len(images) > 0:
        pic = Image.open(
            os.path.join(image_dir, images[rnd.randint(0, len(images) - 1)])
        )

        if pic.size[0] < width:
            pic = pic.resize(
                [width, int(pic.size[1] * (width / pic.size[0]))],
                Image.Resampling.LANCZOS,
            )
        if pic.size[1] < height:
            pic = pic.resize(
                [int(pic.size[0] * (height / pic.size[1])), height],
                Image.Resampling.LANCZOS,
            )

        if pic.size[0] == width:
            x = 0
        else:
            x = rnd.randint(0, pic.size[0] - width)
        if pic.size[1] == height:
            y = 0
        else:
            y = rnd.randint(0, pic.size[1] - height)

        return pic.crop((x, y, x + width, y + height))
    else:
        raise Exception("No images where found in the images folder!")
