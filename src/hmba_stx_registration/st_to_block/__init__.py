"""
Stage 1 – Register spatial-TX cell coordinates to blockface images.

This module computes a 2-D affine transform that maps cell coordinates
(in blockface-pixel units) to the coordinate system of a blockface PNG.

**Outputs** (written to *transforms_out_path*):

* ``{specimen_name}_sptx_to_bf_affine.npy``  – 3×3 affine (NumPy)
* ``qc/{specimen_name}_registration_qc.png`` – 6-panel QC figure

The registration pipeline:
1. Build WM / GM density images from cell-type labels.
2. Rotation search (brute-force) + phase-correlation for translation.
3. Pre-align with the coarse transform.
4. Refine with SimpleITK affine registration (multi-resolution).
5. Chain the coarse and fine transforms → single 3×3 affine.

Public API
----------
- ``register_cells_to_blockface``  – main entry point
- ``create_density_image``         – build density images from cell table
- ``coarse_alignment``             – rotation search + phase-correlation
- ``sitk_affine_refinement``       – SimpleITK affine refinement
- ``apply_affine_to_coordinates``  – apply combined affine to cell coords
- ``save_qc_figure``               – 6-panel QC figure
- ``compute_wm_gm_density``       – helper exposed for testing / reuse
- ``rotation_search``             – brute-force rotation finder
- ``rotate_and_phase_correlate``   – manual rotation + phase-correlation
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.exposure import equalize_hist
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp

from ..utils import generate_random_colormap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# White-matter / grey-matter density
# ---------------------------------------------------------------------------

WM_TYPES = [
    "Oligodendrocyte",
    "Oligodendrocyte precursor",
    "Committed oligodendrocyte precursor",
]

GM_TYPES = [
    "Deep-layer near-projecting",
    "Deep-layer corticothalamic and 6b",
    "Deep-layer intratelencephalic",
    "Upper-layer intratelencephalic",
    "MGE interneuron",
    "CGE interneuron",
    "LAMP5-LHX6 and Chandelier",
]


def compute_wm_gm_density(
    table: pd.DataFrame,
    table_label: str,
    um_per_px: float,
    kernel_size_um: float = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute white-matter and grey-matter density maps.

    Parameters
    ----------
    table : pd.DataFrame
        Cell table with ``center_x``, ``center_y``, and *table_label* columns.
    table_label : str
        Column name containing cell-type labels.
    um_per_px : float
        Microns per pixel.
    kernel_size_um : float
        Gaussian kernel size in µm for density smoothing.

    Returns
    -------
    wm_mask, gm_mask : np.ndarray (bool)
        Binary masks.
    ratio : np.ndarray
        ``(WM − GM) / (WM + GM + ε)``; positive → WM, negative → GM.
    wm_density, gm_density : np.ndarray
        Smoothed density images.
    """
    table = table.copy()
    table["is_wm"] = table[table_label].isin(WM_TYPES).astype(float)
    table["is_gm"] = table[table_label].isin(GM_TYPES).astype(float)

    x_px = (table["center_x"].values / um_per_px).astype(int)
    y_px = (table["center_y"].values / um_per_px).astype(int)
    h, w = y_px.max() + 1, x_px.max() + 1

    wm_img = np.zeros((h, w), dtype=np.float32)
    gm_img = np.zeros((h, w), dtype=np.float32)

    np.add.at(wm_img, (y_px, x_px), table["is_wm"].values)
    np.add.at(gm_img, (y_px, x_px), table["is_gm"].values)

    kernel_px = int(kernel_size_um / um_per_px)
    kernel_px = kernel_px if kernel_px % 2 == 1 else kernel_px + 1

    wm_density = cv2.GaussianBlur(wm_img, (kernel_px, kernel_px), 0)
    gm_density = cv2.GaussianBlur(gm_img, (kernel_px, kernel_px), 0)

    total = wm_density + gm_density + 1e-6
    ratio = (wm_density - gm_density) / total

    wm_mask = ratio > 0
    gm_mask = ratio < 0

    return wm_mask, gm_mask, ratio, wm_density, gm_density


# ---------------------------------------------------------------------------
# Rotation search helpers
# ---------------------------------------------------------------------------

def _evaluate_angle(
    angle: float,
    moving: np.ndarray,
    fixed_on_canvas: np.ndarray,
    canvas_shape: Tuple[int, int],
    canvas_center: np.ndarray,
    moving_center: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """Evaluate a single rotation angle (designed for parallel execution)."""
    angle_rad = np.deg2rad(angle)

    T_to_origin = AffineTransform(translation=-moving_center)
    R = AffineTransform(rotation=-angle_rad)
    T_to_canvas = AffineTransform(translation=canvas_center)

    moving_to_canvas = AffineTransform(
        matrix=T_to_canvas.params @ R.params @ T_to_origin.params,
    )
    moving_on_canvas = warp(
        moving, moving_to_canvas.inverse, output_shape=canvas_shape, preserve_range=True,
    )

    shift, _error, _diffphase = phase_cross_correlation(fixed_on_canvas, moving_on_canvas)

    shifted = np.roll(
        np.roll(moving_on_canvas, int(shift[0]), axis=0),
        int(shift[1]),
        axis=1,
    )
    corr = np.corrcoef(fixed_on_canvas.flatten(), shifted.flatten())[0, 1]

    return angle, shift, corr


def rotation_search(
    fixed: np.ndarray,
    moving: np.ndarray,
    angle_step: int = 5,
    n_workers: Optional[int] = None,
) -> Tuple[AffineTransform, int, Tuple, float]:
    """Brute-force rotation search with phase-correlation translation.

    Parameters
    ----------
    fixed : np.ndarray
        Reference image.
    moving : np.ndarray
        Image to align.
    angle_step : int
        Angular step in degrees.
    n_workers : int, optional
        Number of parallel workers (defaults to CPU count).

    Returns
    -------
    transform : AffineTransform
        Maps *moving* coordinates → *fixed* coordinates.
    best_angle : int
        Best rotation (degrees).
    best_shift : tuple
        Best translation (row, col).
    best_corr : float
        Pearson correlation at best alignment.
    """
    max_dim = int(
        np.ceil(
            np.sqrt(
                fixed.shape[0] ** 2
                + fixed.shape[1] ** 2
                + moving.shape[0] ** 2
                + moving.shape[1] ** 2
            )
        )
    )
    canvas_shape = (max_dim, max_dim)

    fixed_center = np.array([fixed.shape[1], fixed.shape[0]]) / 2
    canvas_center = np.array([canvas_shape[1], canvas_shape[0]]) / 2

    fixed_to_canvas = AffineTransform(translation=canvas_center - fixed_center)
    fixed_on_canvas = warp(
        fixed, fixed_to_canvas.inverse, output_shape=canvas_shape, preserve_range=True,
    )

    moving_center = np.array([moving.shape[1], moving.shape[0]]) / 2

    eval_func = partial(
        _evaluate_angle,
        moving=moving,
        fixed_on_canvas=fixed_on_canvas,
        canvas_shape=canvas_shape,
        canvas_center=canvas_center,
        moving_center=moving_center,
    )

    best_angle = 0
    best_corr = -np.inf
    best_shift: Tuple = (0, 0)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(eval_func, a): a for a in range(0, 360, angle_step)}
        for future in as_completed(futures):
            angle, shift, corr = future.result()
            if corr > best_corr:
                best_corr = corr
                best_angle = angle
                best_shift = shift

    # Build final moving→fixed transform
    angle_rad = np.deg2rad(best_angle)

    T_to_origin = AffineTransform(translation=-moving_center)
    R = AffineTransform(rotation=-angle_rad)
    T_to_canvas = AffineTransform(translation=canvas_center)
    T_shift = AffineTransform(translation=[best_shift[1], best_shift[0]])
    T_to_fixed = AffineTransform(translation=fixed_center - canvas_center)

    moving_to_fixed = AffineTransform(
        matrix=T_to_fixed.params @ T_shift.params @ T_to_canvas.params @ R.params @ T_to_origin.params,
    )

    return moving_to_fixed, best_angle, best_shift, best_corr


def rotate_and_phase_correlate(
    fixed: np.ndarray,
    moving: np.ndarray,
    angle_degrees: float,
    upsample_factor: int = 10,
) -> Tuple[AffineTransform, np.ndarray, Tuple[float, float]]:
    """Rotate by a known angle and refine translation via phase correlation.

    Parameters
    ----------
    fixed : np.ndarray
        Reference image (target).
    moving : np.ndarray
        Image to be transformed (source).
    angle_degrees : float
        Rotation angle in degrees (counter-clockwise positive).
    upsample_factor : int
        Sub-pixel precision factor for phase correlation.

    Returns
    -------
    affine : AffineTransform
        Combined rotation + translation (moving → fixed).
    aligned : np.ndarray
        Transformed *moving* image aligned to *fixed*.
    shift : tuple of float
        Translation ``(row, col)`` found by phase correlation.
    """
    moving_center = np.array([moving.shape[1] / 2, moving.shape[0] / 2])
    fixed_center = np.array([fixed.shape[1] / 2, fixed.shape[0] / 2])

    angle_rad = np.deg2rad(angle_degrees)

    T_to_origin = AffineTransform(translation=-moving_center)
    R = AffineTransform(rotation=-angle_rad)
    T_to_fixed_center = AffineTransform(translation=fixed_center)

    rotation_transform = AffineTransform(
        matrix=T_to_fixed_center.params @ R.params @ T_to_origin.params,
    )

    rotated = warp(
        moving,
        rotation_transform.inverse,
        output_shape=fixed.shape,
        mode="constant",
        cval=0,
    )

    shift, _error, _diffphase = phase_cross_correlation(
        fixed.astype(np.float64),
        rotated.astype(np.float64),
        upsample_factor=upsample_factor,
    )

    T_shift = AffineTransform(translation=np.array([shift[1], shift[0]]))

    combined_affine = AffineTransform(
        matrix=T_shift.params @ rotation_transform.params,
    )

    aligned = warp(
        moving,
        combined_affine.inverse,
        output_shape=fixed.shape,
        mode="constant",
        cval=0,
    )

    return combined_affine, aligned.astype(moving.dtype), tuple(shift)


# ---------------------------------------------------------------------------
# Sub-functions (extracted from register_cells_to_blockface)
# ---------------------------------------------------------------------------

def create_density_image(
    cells_table: pd.DataFrame,
    blockface_img_path: Path,
    table_label: str = "supercluster_term_name",
    um_per_px: float = 20,
    gamma: float = 1.0,
    kernel_px: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build WM/GM density and blockface images for registration.

    Parameters
    ----------
    cells_table : pd.DataFrame
        Cell table with ``center_x``, ``center_y``, and *table_label*.
    blockface_img_path : Path
        Path to the blockface PNG (RGBA expected).
    table_label : str
        Column containing cell-type labels.
    um_per_px : float
        Microns per pixel.
    gamma : float
        Gamma correction exponent applied after histogram equalisation.
    kernel_px : int
        Gaussian blur kernel size (pixels) for the density image.

    Returns
    -------
    cells_img : np.ndarray
        Processed density image (histogram-equalised, gamma-corrected).
    mask : np.ndarray
        Processed blockface grayscale image.
    coords_h : np.ndarray
        Homogeneous cell coordinates (N×3) in pixel units.
    """
    _wm_mask, _gm_mask, ratio, _wm_d, _gm_d = compute_wm_gm_density(
        cells_table, table_label, um_per_px, kernel_size_um=200,
    )
    image = plt.imread(str(blockface_img_path))
    mask = image[:, :, 0] * image[:, :, 3]

    cells_img = ratio + 2
    cells_img[ratio == 0] = 0
    cells_img /= cells_img.max()
    cells_img = cv2.GaussianBlur(cells_img, (kernel_px, kernel_px), 0)

    cells_img = equalize_hist(cells_img, mask=cells_img > 0) ** gamma
    mask = equalize_hist(mask, mask=mask > 0) ** gamma

    coords = cells_table[["center_x", "center_y"]].values / um_per_px
    coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])

    return cells_img, mask, coords_h


def coarse_alignment(
    mask: np.ndarray,
    cells_img: np.ndarray,
    rotation: Optional[float] = None,
) -> Tuple[AffineTransform, np.ndarray]:
    """Find the initial rigid alignment (rotation + translation).

    When *rotation* is ``None`` a brute-force search over 360° is
    performed; otherwise the given angle is used directly.

    Parameters
    ----------
    mask : np.ndarray
        Reference (blockface) image.
    cells_img : np.ndarray
        Moving (density) image.
    rotation : float, optional
        If provided, skip the brute-force search and use this angle.

    Returns
    -------
    init_aff : AffineTransform
        Coarse moving → fixed transform.
    cells_img_prealigned : np.ndarray
        *cells_img* warped onto *mask* coordinates.
    """
    if rotation is None:
        init_aff, best_angle, best_shift, best_corr = rotation_search(
            mask, cells_img, angle_step=5,
        )
        logger.info(
            "Best rotation: %d°, shift: %s, corr: %.4f",
            best_angle, best_shift, best_corr,
        )
    else:
        init_aff, _, best_shift = rotate_and_phase_correlate(
            mask, cells_img, rotation, upsample_factor=10,
        )
        logger.info("Applied rotation: %s°, shift: %s", rotation, best_shift)

    cells_img_prealigned = warp(
        cells_img, init_aff.inverse, output_shape=mask.shape,
    ).astype(np.float32)

    return init_aff, cells_img_prealigned


def sitk_affine_refinement(
    mask: np.ndarray,
    cells_img_prealigned: np.ndarray,
    init_aff: AffineTransform,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine alignment with SimpleITK affine registration.

    Parameters
    ----------
    mask : np.ndarray
        Fixed (blockface) image.
    cells_img_prealigned : np.ndarray
        Moving image after coarse alignment.
    init_aff : AffineTransform
        Coarse transform (used to chain with the refinement).

    Returns
    -------
    combined_aff : np.ndarray
        3×3 combined affine (coarse ∘ fine) in pixel coordinates.
    cells_img_warped : np.ndarray
        Moving image resampled with the fine transform.
    """
    fixed_sitk = sitk.GetImageFromArray(mask.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(cells_img_prealigned.astype(np.float32))

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-6,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.AffineTransform(2)
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed_sitk, moving_sitk)

    cells_img_warped = sitk.GetArrayFromImage(
        sitk.Resample(moving_sitk, fixed_sitk, final_transform, sitk.sitkLinear, 0.0),
    )

    # Extract affine from (possibly composite) SimpleITK transform
    if isinstance(final_transform, sitk.CompositeTransform):
        n_transforms = final_transform.GetNumberOfTransforms()
        affine_tfm = None
        for i in range(n_transforms):
            tfm = final_transform.GetNthTransform(i)
            if isinstance(tfm, sitk.AffineTransform):
                affine_tfm = tfm
                break
        if affine_tfm is None:
            affine_tfm = sitk.AffineTransform(2)
            affine_tfm.SetMatrix(final_transform.GetNthTransform(0).GetMatrix())
            affine_tfm.SetTranslation(final_transform.GetNthTransform(0).GetTranslation())
    else:
        affine_tfm = final_transform

    M = np.array(affine_tfm.GetMatrix(), dtype=float).reshape(2, 2)
    T = np.array(affine_tfm.GetTranslation(), dtype=float)

    results_aff = np.eye(3, dtype=float)
    results_aff[:2, :2] = M
    results_aff[:2, 2] = T
    results_aff = np.linalg.inv(results_aff)
    combined_aff = results_aff @ init_aff.params

    return combined_aff, cells_img_warped


def apply_affine_to_coordinates(
    cells_table: pd.DataFrame,
    coords_h: np.ndarray,
    init_aff: AffineTransform,
    combined_aff: np.ndarray,
    specimen_name: str,
    transforms_out_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project cell coordinates through the coarse and combined affines.

    Also persists the combined affine to disk and annotates *cells_table*
    with blockface-pixel columns.

    Parameters
    ----------
    cells_table : pd.DataFrame
        Cell table (modified in place with ``x_blockface_px``,
        ``y_blockface_px`` columns).
    coords_h : np.ndarray
        Homogeneous coordinates (N×3) in pixel units.
    init_aff : AffineTransform
        Coarse transform.
    combined_aff : np.ndarray
        3×3 combined affine.
    specimen_name : str
        Identifier used for the output filename.
    transforms_out_path : Path
        Directory for the ``.npy`` affine file.

    Returns
    -------
    coords_affine : np.ndarray
        Coordinates after coarse transform only (N×2).
    coords_warped : np.ndarray
        Coordinates after the full combined affine (N×2).
    """
    coords_affine = (init_aff.params @ coords_h.T).T[:, :2]
    coords_warped = (combined_aff @ coords_h.T).T[:, :2]

    transforms_out_path.mkdir(parents=True, exist_ok=True)
    np.save(transforms_out_path / f"{specimen_name}_sptx_to_bf_affine.npy", combined_aff)

    cells_table["x_blockface_px"] = coords_warped[:, 0]
    cells_table["y_blockface_px"] = coords_warped[:, 1]

    return coords_affine, coords_warped


def save_qc_figure(
    cells_img: np.ndarray,
    cells_img_prealigned: np.ndarray,
    cells_img_warped: np.ndarray,
    mask: np.ndarray,
    coords_affine: np.ndarray,
    coords_warped: np.ndarray,
    cells_table: pd.DataFrame,
    table_label: str,
    specimen_name: str,
    qc_path: Path,
) -> None:
    """Generate and save a 6-panel QC figure.

    Parameters
    ----------
    cells_img : np.ndarray
        Original density image (microscope coordinates).
    cells_img_prealigned : np.ndarray
        Density image after coarse alignment.
    cells_img_warped : np.ndarray
        Density image after full affine alignment.
    mask : np.ndarray
        Blockface grayscale image.
    coords_affine : np.ndarray
        Coordinates after coarse transform (N×2).
    coords_warped : np.ndarray
        Coordinates after full combined affine (N×2).
    cells_table : pd.DataFrame
        Cell table (used for label colours).
    table_label : str
        Column containing cell-type labels.
    specimen_name : str
        Identifier used for the output filename.
    qc_path : Path
        Directory for the QC PNG.
    """
    qc_path.mkdir(parents=True, exist_ok=True)

    label_cmap = generate_random_colormap(cells_table[table_label].unique(), seed=44)
    label_color = cells_table[table_label].map(label_cmap)

    fig, ax = plt.subplots(1, 6, figsize=(30, 5))
    ax[0].imshow(cells_img, cmap="gray")
    ax[0].set_title("Spatial WM/GM density image (microscope coords)")
    ax[1].imshow(cells_img_prealigned, cmap="gray")
    ax[1].scatter(
        coords_affine[::100, 0], coords_affine[::100, 1],
        s=1, c=label_color[::100], alpha=0.5,
    )
    ax[1].set_title("Spatial WM/GM density image (rot+shift)")
    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title("Blockface image (grayscale)")
    ax[3].imshow(cells_img_warped, cmap="gray")
    ax[3].set_title("Spatial WM/GM density image (affine)")
    ax[4].imshow(cells_img_warped, cmap="gray")
    ax[4].scatter(
        coords_warped[::100, 0], coords_warped[::100, 1],
        s=1, c=label_color[::100], alpha=0.5,
    )
    ax[4].set_title("Aligned cells over WM/GM density (affine)")
    ax[5].imshow(mask, cmap="gray")
    ax[5].scatter(
        coords_warped[::100, 0], coords_warped[::100, 1],
        s=1, c=label_color[::100], alpha=0.5,
    )
    ax[5].set_title("Aligned cells over blockface")
    for a in ax:
        a.axis("off")
    plt.savefig(qc_path / f"{specimen_name}_registration_qc.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main registration function
# ---------------------------------------------------------------------------

def register_cells_to_blockface(
    cells_table: pd.DataFrame,
    blockface_img_path: Path,
    specimen_name: str,
    transforms_out_path: Path,
    qc_path: Path,
    table_label: str = "supercluster_term_name",
    um_per_px: float = 20,
    rotation: Optional[float] = None,
    gamma: float = 1.0,
    kernel_px: int = 5,
) -> np.ndarray:
    """Register spatial-TX cells to a blockface image and save outputs.

    The function:
    1. Builds a WM/GM density image from cell-type labels.
    2. Finds the best rotation (or uses a provided one) via phase correlation.
    3. Refines the alignment with a SimpleITK affine registration.
    4. Applies the combined affine to cell coordinates and saves it.
    5. Generates a 6-panel QC figure.

    Parameters
    ----------
    cells_table : pd.DataFrame
        Cell table with ``center_x``, ``center_y``, and *table_label*.
    blockface_img_path : Path
        Path to the blockface PNG (RGBA expected).
    specimen_name : str
        Identifier used for output filenames.
    transforms_out_path : Path
        Directory for the ``.npy`` affine file.
    qc_path : Path
        Directory for the QC PNG.
    table_label : str
        Column containing cell-type labels.
    um_per_px : float
        Microns per pixel.
    rotation : float, optional
        If provided, skip the brute-force search and use this angle.
    gamma : float
        Gamma correction exponent applied after histogram equalisation.
    kernel_px : int
        Gaussian blur kernel size (pixels) for the density image.

    Returns
    -------
    np.ndarray
        The combined 3×3 affine (sptx pixel → blockface pixel).
    """
    # 1. density images
    cells_img, mask, coords_h = create_density_image(
        cells_table, blockface_img_path, table_label, um_per_px, gamma, kernel_px,
    )

    # 2. coarse alignment
    init_aff, cells_img_prealigned = coarse_alignment(mask, cells_img, rotation)

    # 3. SimpleITK affine refinement
    combined_aff, cells_img_warped = sitk_affine_refinement(
        mask, cells_img_prealigned, init_aff,
    )

    # 4. apply combined affine to coordinates
    coords_affine, coords_warped = apply_affine_to_coordinates(
        cells_table, coords_h, init_aff, combined_aff,
        specimen_name, transforms_out_path,
    )

    # 5. QC figure
    save_qc_figure(
        cells_img, cells_img_prealigned, cells_img_warped, mask,
        coords_affine, coords_warped,
        cells_table, table_label, specimen_name, qc_path,
    )

    logger.info("Saved blockface registration for %s", specimen_name)
    return combined_aff
