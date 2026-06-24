"""
Translational registration of blockface images using a Photoshop-style
"magic wand" selection of the out-of-focus metallic background.

The microscope rig itself (the metallic hardware visible at the corners
of each blockface frame) is essentially stationary in the world but
shifts a few pixels between captures because of small camera-position
changes. Because the metallic region is dim, low-contrast, and roughly
identical across frames, it is an excellent fiducial for *pure
translation* recovery via phase correlation.

Workflow exposed by this module:

1. :func:`magic_wand_mask` – build a binary mask of pixels connected to
   a user-supplied seed whose colour is within ``tolerance`` of the
   seed colour (mimicking Photoshop's *contiguous magic wand*).
2. :func:`register_translations_via_magic_wand` – open every image in a
   list, compute the mask once on a reference image, then use masked
   phase correlation to estimate the integer/sub-pixel translation that
   maps each remaining image onto the reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation


PathLike = Union[str, Path]


def magic_wand_mask(
    image_bgr: np.ndarray,
    seed_xy: Tuple[int, int],
    tolerance: float = 20.0,
    *,
    connectivity: int = 8,
    contiguous: bool = True,
) -> np.ndarray:
    """Magic-wand-style selection rooted at *seed_xy*.

    Mirrors Photoshop's magic wand: every pixel reachable from the seed
    whose colour is within ``±tolerance`` (per channel) of the *seed
    pixel*'s colour is included.

    Parameters
    ----------
    image_bgr
        Input image (``uint8``). May be 2-D grayscale or 3-channel BGR.
    seed_xy
        ``(x, y)`` seed coordinate, in image pixel coordinates.
    tolerance
        Per-channel colour-distance threshold (Photoshop "Tolerance").
    connectivity
        ``4`` or ``8``-connected flood fill.
    contiguous
        If ``True`` (default), only pixels reachable from the seed are
        selected (Photoshop's *Contiguous* checkbox enabled). If
        ``False``, every pixel in the image whose colour is within
        ``tolerance`` of the seed colour is selected.

    Returns
    -------
    np.ndarray
        ``uint8`` binary mask (``255`` inside selection, ``0`` outside),
        same height/width as *image_bgr*.
    """
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be uint8")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    h, w = image_bgr.shape[:2]
    x, y = int(seed_xy[0]), int(seed_xy[1])
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"seed_xy {seed_xy} is outside image of shape {(h, w)}")

    if not contiguous:
        seed_color = image_bgr[y, x].astype(np.int16)
        if image_bgr.ndim == 2:
            diff = np.abs(image_bgr.astype(np.int16) - seed_color)
        else:
            diff = np.max(
                np.abs(image_bgr.astype(np.int16) - seed_color[None, None, :]),
                axis=2,
            )
        return (diff <= tolerance).astype(np.uint8) * 255

    # Contiguous flood fill mimicking Photoshop's magic wand.
    img = image_bgr.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    n_channels = 1 if img.ndim == 2 else img.shape[2]
    lo = (float(tolerance),) * n_channels
    up = (float(tolerance),) * n_channels

    flags = (
        connectivity
        | cv2.FLOODFILL_FIXED_RANGE   # compare to seed (Photoshop) not neighbours
        | cv2.FLOODFILL_MASK_ONLY     # only fill the mask, leave img untouched
        | (255 << 8)                  # mask new-value = 255
    )
    cv2.floodFill(img, flood_mask, (x, y), 0, lo, up, flags)
    # flood_mask is already 0/255 because of the (255 << 8) flag.
    return flood_mask[1:-1, 1:-1].copy()


def _load_bgr(path: Path) -> np.ndarray:
    """Load *path* as a 3-channel BGR ``uint8`` array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.dtype != np.uint8:
        info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
        if info is not None:
            img = (img.astype(np.float32) * (255.0 / info.max)).clip(0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def register_translations_via_magic_wand(
    image_paths: List[PathLike],
    seed_xy: Tuple[int, int],
    tolerance: float = 20.0,
    *,
    reference_index: int = 0,
    upsample_factor: int = 10,
    return_aligned: bool = False,
) -> Dict[str, object]:
    """Estimate translation-only alignments between *image_paths*.

    A mask of the metallic background is selected once on the reference
    image (``image_paths[reference_index]``) using
    :func:`magic_wand_mask`. The same mask is then used for masked phase
    correlation against every other image, recovering the ``(dy, dx)``
    shift that maps each *moving* image onto the reference.

    The metallic region is assumed to be approximately stationary in
    the world (only camera-position jitter), so reusing the reference
    mask on each moving image is a reasonable simplification.

    Parameters
    ----------
    image_paths
        Ordered list of image paths to co-register.
    seed_xy
        ``(x, y)`` seed coordinate (in pixel coordinates of the
        reference image) where the magic wand is applied — pick a point
        inside the out-of-focus metallic background.
    tolerance
        Photoshop-style per-channel colour tolerance for the magic wand.
    reference_index
        Index into *image_paths* of the image used as the registration
        reference. Defaults to the first image.
    upsample_factor
        Sub-pixel upsampling factor passed to
        :func:`skimage.registration.phase_cross_correlation`.
        ``1`` = pixel precision; ``10`` = 0.1-pixel precision.
    return_aligned
        If ``True``, also load and shift each moving image and return
        them in the result dict. Aligned images are returned as the
        original 3-channel BGR ``uint8`` arrays (same colour layout as
        :func:`cv2.imread`).

    Returns
    -------
    dict
        ``{"reference_path": Path, "mask": np.ndarray,
        "shifts": {str(path): (dy, dx)},
        "errors": {str(path): float},
        "aligned": {str(path): np.ndarray} or None}``.

        Each ``(dy, dx)`` is the translation that, when applied to the
        moving image (``scipy.ndimage.shift``), aligns it onto the
        reference. The reference path itself is included with a zero
        shift.

    Raises
    ------
    ValueError
        If *image_paths* is empty, *reference_index* is out of range,
        or the magic wand selects an empty mask.
    """
    if len(image_paths) == 0:
        raise ValueError("image_paths must contain at least one path")
    if not (0 <= reference_index < len(image_paths)):
        raise ValueError(
            f"reference_index={reference_index} is out of range for "
            f"{len(image_paths)} images"
        )

    paths = [Path(p) for p in image_paths]
    ref_path = paths[reference_index]

    ref_bgr = _load_bgr(ref_path)
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    mask = magic_wand_mask(ref_bgr, seed_xy, tolerance=tolerance)
    if mask.sum() == 0:
        raise ValueError(
            f"Magic wand at {seed_xy} with tolerance={tolerance} produced an "
            f"empty mask on {ref_path.name}"
        )
    mask_bool = mask > 0

    shifts: Dict[str, Tuple[float, float]] = {str(ref_path): (0.0, 0.0)}
    errors: Dict[str, float] = {str(ref_path): 0.0}
    aligned: Optional[Dict[str, np.ndarray]] = {} if return_aligned else None
    if return_aligned:
        aligned[str(ref_path)] = ref_bgr

    for i, p in enumerate(paths):
        if i == reference_index:
            continue
        mov_bgr = _load_bgr(p)
        mov_gray = cv2.cvtColor(mov_bgr, cv2.COLOR_BGR2GRAY)
        if mov_gray.shape != ref_gray.shape:
            raise ValueError(
                f"Image shape mismatch: reference {ref_gray.shape} vs "
                f"{p.name} {mov_gray.shape}"
            )

        shift_yx, error, _ = phase_cross_correlation(
            ref_gray,
            mov_gray,
            reference_mask=mask_bool,
            moving_mask=mask_bool,
            upsample_factor=upsample_factor,
        )
        dy, dx = float(shift_yx[0]), float(shift_yx[1])
        shifts[str(p)] = (dy, dx)
        errors[str(p)] = float(error) if np.isfinite(error) else float("nan")

        if return_aligned:
            # Shift each colour channel independently (no shift on axis 2).
            aligned[str(p)] = ndi_shift(
                mov_bgr, shift=(dy, dx, 0), order=1, mode="constant", cval=0
            ).astype(np.uint8)

    return {
        "reference_path": ref_path,
        "mask": mask,
        "shifts": shifts,
        "errors": errors,
        "aligned": aligned,
    }
