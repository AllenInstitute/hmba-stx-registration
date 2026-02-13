"""
SVG parsing utilities for extracting affine transforms from SVG files.

Adapted from utils.svg_transform for use in the HMBA STX registration pipeline.
These functions parse SVG files that encode blockface image positions on slab
images, extracting the affine transformations needed to map blockface pixel
coordinates to slab pixel coordinates.
"""

import math
import re
from pathlib import Path
from typing import Dict

import numpy as np
import xmltodict


# ---------------------------------------------------------------------------
# SVG file I/O
# ---------------------------------------------------------------------------

def load_svg_to_dict(file_path: Path) -> dict:
    """Parse an SVG file into a nested dictionary via *xmltodict*.

    Parameters
    ----------
    file_path : Path
        Path to the SVG file.

    Returns
    -------
    dict
        Nested dictionary representation of the SVG XML.
    """
    with open(file_path, "r") as fh:
        return xmltodict.parse(fh.read())


# ---------------------------------------------------------------------------
# SVG transform string → 3×3 matrix
# ---------------------------------------------------------------------------

def parse_svg_transform(transform_str: str) -> np.ndarray:
    """Convert an SVG ``transform`` attribute string to a 3×3 affine matrix.

    Supports: ``matrix``, ``translate``, ``scale``, ``rotate``,
    ``skewX``, ``skewY``.  Multiple transforms are composed left-to-right
    (i.e. in the order they appear in the string).

    Parameters
    ----------
    transform_str : str
        The value of the SVG ``transform`` attribute,
        e.g. ``"translate(10,20) rotate(45)"``.

    Returns
    -------
    np.ndarray
        3×3 homogeneous affine matrix.
    """
    def _parse_args(s: str):
        return list(map(float, re.findall(r"-?[\d.]+(?:e[-+]?\d+)?", s)))

    def _rotate(angle_deg, cx=0, cy=0):
        theta = math.radians(angle_deg)
        c, s = math.cos(theta), math.sin(theta)
        if cx == 0 and cy == 0:
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
        return T2 @ R @ T1

    def _translate(tx, ty=0):
        return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    def _scale(sx, sy=None):
        sy = sx if sy is None else sy
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    def _skewX(angle_deg):
        return np.array([[1, math.tan(math.radians(angle_deg)), 0],
                         [0, 1, 0], [0, 0, 1]])

    def _skewY(angle_deg):
        return np.array([[1, 0, 0],
                         [math.tan(math.radians(angle_deg)), 1, 0],
                         [0, 0, 1]])

    def _matrix(a, b, c, d, e, f):
        return np.array([[a, c, e], [b, d, f], [0, 0, 1]])

    _builders = {
        "matrix": _matrix,
        "translate": _translate,
        "scale": _scale,
        "rotate": _rotate,
        "skewX": _skewX,
        "skewY": _skewY,
    }

    result = np.eye(3)
    for match in re.finditer(
        r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]+)\)",
        transform_str,
    ):
        name, args_str = match.groups()
        args = _parse_args(args_str)
        builder = _builders.get(name)
        if builder is not None:
            result = result @ builder(*args)

    return result


# ---------------------------------------------------------------------------
# Affine helpers
# ---------------------------------------------------------------------------

def get_scale_affine(
    x: float,
    y: float,
    affine: np.ndarray | None = None,
) -> np.ndarray:
    """Return a 3×3 scale matrix (or modify *affine* in-place).

    Parameters
    ----------
    x, y : float
        Scale factors along x and y.
    affine : np.ndarray, optional
        If provided, its diagonal entries are overwritten.

    Returns
    -------
    np.ndarray
        3×3 affine with the requested scale.
    """
    if affine is None:
        affine = np.eye(3)
    else:
        assert affine.shape == (3, 3)
    affine[0, 0] = x
    affine[1, 1] = y
    return affine


def get_translation_affine(
    x: float,
    y: float,
    affine: np.ndarray | None = None,
) -> np.ndarray:
    """Return a 3×3 translation matrix.

    Parameters
    ----------
    x, y : float
        Translation offsets.
    affine : np.ndarray, optional
        If provided, its translation entries are overwritten.

    Returns
    -------
    np.ndarray
        3×3 affine with the requested translation.
    """
    if affine is None:
        affine = np.eye(3)
    else:
        assert affine.shape == (3, 3)
    affine[0, 2] = x
    affine[1, 2] = y
    return affine


def aggregate_affine(
    downsample_factor: float,
    x: float,
    y: float,
    scale_x: float,
    scale_y: float,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    """Compose translation, scale, downsample, and SVG transform into one affine.

    The resulting matrix maps *source image pixels* (at native resolution)
    into the SVG/slab coordinate system:

        ``transform_matrix @ translation(x, y) @ scale(scale_x, scale_y) @ scale(downsample)``

    Parameters
    ----------
    downsample_factor : float
        Uniform downsample applied first (usually 1).
    x, y : float
        SVG ``x`` / ``y`` position of the image element.
    scale_x, scale_y : float
        Width/height ratio between the SVG element size and the source image
        pixel dimensions.
    transform_matrix : np.ndarray
        3×3 matrix parsed from the SVG ``transform`` attribute.

    Returns
    -------
    np.ndarray
        Composed 3×3 affine matrix.
    """
    translation = get_translation_affine(x, y)
    scale = get_scale_affine(scale_x, scale_y)
    downsample = get_scale_affine(downsample_factor, downsample_factor)
    return transform_matrix @ translation @ scale @ downsample
