"""
Helper functions for generating colormaps for concepts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb


def make_fixed_colors(nb_colors: int) -> list[list[float]]:
    """
    Generate a fixed set of colors evenly distributed across the specified number of colors.

    Args:
        nb_colors (int): Number of colors

    Returns:
        List[List[float]]: List of colors in RGB format
    """
    colors = []
    for i in range(nb_colors):
        hue = i / nb_colors
        rgb = hsv_to_rgb([hue, 1, 1]).tolist()
        colors.append(rgb)
    return colors


def make_random_colors(nb_colors: int) -> list[list[float]]:
    """
    Generate random colors.

    Args:
        nb_colors (int): Number of colors

    Returns:
        List[List[float]]: List of colors in RGB format
    """
    return hsv_to_rgb([[hue, 1, 1] for hue in np.random.rand(nb_colors)]).tolist()


def display_color_gradients(colors_list: list[dict]):
    """
    Display a gradient for each color in the list.

    Args:
        colors_list (List[dict]): List of colors in RGB format
    """
    plt.figure(figsize=(10, 2))
    for i, color in enumerate(colors_list):
        gradient = np.zeros((1, 256, 4))
        gradient[:, :, :3] = color[:3]  # Set RGB channels
        gradient[:, :, 3] = np.linspace(0, 1, 256)  # Vary alpha channel
        plt.imshow(gradient, aspect="auto", extent=[i, i + 1, 0, 1])

    plt.xlim(0, len(colors_list))
    plt.axis("off")
    plt.show()
