"""
Block-image segmentation helpers and interactive processing pipeline.

These utilities are used by the block-registration workflow to:

1. Find unique block image paths inside a slab directory.
2. Load and exposure-normalize blockface images.
3. Segment the green OCT/resin region (auto or SAM-assisted).
4. Crop to the OCT bounding box.
5. Segment the embedded tissue (SAM-assisted with a user-drawn bbox).
6. Save the resulting RGBA cutout.

The combining function :func:`process_slab` runs the whole pipeline for
every block specimen in a slab folder.

The functions that call SAM accept a ``predictor`` argument so the
``segment-anything`` package is **not** an import-time dependency of
this module — callers wire up SAM themselves (typically in a notebook).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def normalize_exposure_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L-channel of a BGR image (exposure normalization)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Lc = clahe.apply(L)
    lab = cv2.merge([Lc, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(x_min, y_min, x_max, y_max)`` for non-zero pixels in *mask*."""
    if mask.ndim != 2:
        raise ValueError("Input must be a 2D binary mask")

    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    return (int(x_indices.min()), int(y_indices.min()),
            int(x_indices.max()), int(y_indices.max()))


def crop_image(
    image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]
) -> Optional[np.ndarray]:
    """Crop *image* to *bbox* ``(x_min, y_min, x_max, y_max)``."""
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max + 1, x_min:x_max + 1]


def rgba_from_rgb_and_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build an RGBA image whose alpha is *mask*."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (H, W, 3)")
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D binary array")

    rgba = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = image
    rgba[..., 3] = mask
    return rgba


# ---------------------------------------------------------------------------
# OCT auto-segmentation (fallback when no user annotation is supplied)
# ---------------------------------------------------------------------------

def segment_green_oct(img_bgr: np.ndarray) -> np.ndarray:
    """Segment the green OCT/resin region of *img_bgr*.

    Uses Otsu on the inverted ``a`` channel of LAB (green has low ``a``),
    with a k-means fallback when the result is implausibly small. Output
    is post-processed (open/close, largest connected component, hole fill).

    Parameters
    ----------
    img_bgr
        OpenCV BGR image (``np.uint8``).

    Returns
    -------
    np.ndarray
        ``uint8`` binary mask (``255`` inside OCT, ``0`` outside).
    """
    L2, A2, B2 = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))
    A_blur = cv2.GaussianBlur(A2, (5, 5), 0)
    _, mask = cv2.threshold(
        cv2.bitwise_not(A_blur), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    if np.count_nonzero(mask) < 0.01 * mask.size:
        ab = np.stack([A2.reshape(-1), B2.reshape(-1)], axis=1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 3
        _, labels, centers = cv2.kmeans(ab, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        green_idx = int(np.argmin(centers[:, 0]))
        mask = (labels.reshape(A2.shape) == green_idx).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m2 = np.zeros_like(mask)
        m2[labels == idx] = 255
        mask = m2

    h, w = mask.shape
    ff = mask.copy()
    flood_buf = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_buf, (0, 0), 255)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(ff))

    return mask


# ---------------------------------------------------------------------------
# Interactive annotation
# ---------------------------------------------------------------------------

def annotate_two_points(
    image_bgr: np.ndarray,
    window_name: str = "Annotate Points",
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Collect exactly two click points from the user via an OpenCV window.

    Left-click adds a point. Press *Enter* to confirm, *Esc* to cancel.

    Returns
    -------
    tuple or None
        ``((x0, y0), (x1, y1))`` on confirmation, ``None`` on cancel
        or if the user did not provide exactly two points.
    """
    disp = image_bgr.copy()
    points: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal disp, points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)
    cv2.moveWindow(window_name, 100, 100)
    cv2.imshow(window_name, disp)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, disp)
        k = cv2.waitKey(33) & 0xFF
        if k in (13, 10):  # Enter
            break
        elif k == 27:  # Esc
            points = []
            break

    cv2.destroyWindow(window_name)
    if len(points) != 2:
        return None
    return points[0], points[1]


# ---------------------------------------------------------------------------
# Pipeline building blocks
# ---------------------------------------------------------------------------

def find_block_image_paths(slab_path: Path, ext: str = ".tif") -> List[Path]:
    """Find unique block image paths in a slab directory.

    Filters files in *slab_path* with extension *ext* to those whose
    specimen name (the portion before the first ``_``) has more than
    five dot-separated components — the block naming convention —
    keeping only the first occurrence per specimen.
    """
    img_paths = list(slab_path.glob(f"*{ext}"))
    img_paths = [p for p in img_paths if len(p.name.split("_")[0].split(".")) > 5]
    unique: dict = {}
    for p in img_paths:
        specimen_id = p.name.split("_")[0]
        if specimen_id not in unique:
            unique[specimen_id] = p
    return list(unique.values())


def load_normalized_image(img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load *img_path* and return ``(bgr, rgb)`` after LAB exposure normalization."""
    bgr = cv2.imread(str(img_path))
    bgr = normalize_exposure_lab(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


def sam_mask_from_bbox(
    predictor,
    rgb_image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mask_index: int = 2,
) -> np.ndarray:
    """Run a SAM ``predictor`` with a bounding-box prompt; return one mask.

    The predictor must already be a SAM ``SamPredictor`` (segment-anything).
    *mask_index* selects which of the multi-mask outputs to return.
    """
    predictor.set_image(rgb_image)
    masks, _, _ = predictor.predict(
        box=np.array(bbox)[None, :],
        multimask_output=True,
    )
    return (masks[mask_index] * 255).astype(np.uint8)


def get_oct_mask(
    predictor,
    bgr_image: np.ndarray,
    rgb_image: np.ndarray,
    annotate: bool = True,
) -> np.ndarray:
    """Return an OCT mask, either annotated via SAM or auto-segmented.

    When *annotate* is true, prompts the user for two click points
    defining a SAM bounding box around the OCT region. If the user
    cancels (no points), falls back to :func:`segment_green_oct`.
    """
    if annotate:
        points = annotate_two_points(bgr_image)
        if points is not None:
            bbox = (points[0][0], points[0][1], points[1][0], points[1][1])
            return sam_mask_from_bbox(predictor, rgb_image, bbox)
    return segment_green_oct(bgr_image)


def crop_to_oct(
    bgr_image: np.ndarray, oct_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop ``bgr_image``, its RGB version, and ``oct_mask`` to the OCT bbox."""
    bbox = get_bbox_from_mask(oct_mask)
    bgr_cropped = crop_image(bgr_image, bbox)
    rgb_cropped = cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2RGB)
    oct_cropped = crop_image(oct_mask, bbox)
    return bgr_cropped, rgb_cropped, oct_cropped


def show_annotation_qc(
    rgb_cropped: np.ndarray,
    oct_cropped: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> None:
    """Display the cropped image with the OCT overlay and user bbox."""
    x0, y0, x1, y1 = bbox
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                         linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    plt.imshow(rgb_cropped)
    plt.imshow(oct_cropped, cmap='rainbow', alpha=0.5)
    plt.axis('off')
    plt.show()


def segment_tissue(
    predictor,
    rgb_cropped: np.ndarray,
    bgr_cropped: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Prompt the user for a tissue bbox, then run SAM to mask the tissue.

    Returns ``(mask, bbox)`` or ``(None, None)`` if annotation was cancelled.
    """
    points = annotate_two_points(bgr_cropped)
    if points is None:
        return None, None
    bbox = (points[0][0], points[0][1], points[1][0], points[1][1])
    mask = sam_mask_from_bbox(predictor, rgb_cropped, bbox)
    return mask, bbox


def save_rgba_image(rgba_image: np.ndarray, outpath: Path, specimen_name: str) -> Path:
    """Save *rgba_image* to ``outpath/{specimen_name}.png`` and return the path."""
    out_img_path = outpath / f"{specimen_name}.png"
    Image.fromarray(rgba_image).save(out_img_path)
    return out_img_path


# ---------------------------------------------------------------------------
# Combining function
# ---------------------------------------------------------------------------

def process_slab(
    slab_path: Path,
    predictor,
    annotate_oct_mask: bool = True,
    show_qc: bool = True,
    ext: str = ".tif",
) -> List[Path]:
    """Segment + crop + save every block specimen image in a slab directory.

    For each block image in *slab_path*:

    1. Load and exposure-normalize the image.
    2. Get an OCT mask (interactive SAM or auto).
    3. Crop the image and the OCT mask to the OCT bounding box.
    4. Interactively segment the tissue inside the cropped image with SAM.
    5. Save the RGBA tissue cutout to ``slab_path/processed/{specimen}.png``.

    Specimens whose output already exists are skipped (mirrors the original
    notebook's behaviour: items with more than 6 dot-separated name
    components are deduped against the processed folder; 6-component
    "block" names are always reprocessed).

    Parameters
    ----------
    slab_path
        Directory containing the slab's block images.
    predictor
        A ``segment_anything.SamPredictor`` instance.
    annotate_oct_mask
        If true, prompt the user for an OCT bbox via SAM. Otherwise use
        the automatic green-OCT segmenter.
    show_qc
        If true, display matplotlib QC figures for each specimen.
    ext
        Image extension to look for in *slab_path*.

    Returns
    -------
    list of Path
        Paths of the newly written PNGs (skipped specimens are excluded).
    """
    slab_path = Path(slab_path)
    outpath = slab_path / "processed"
    outpath.mkdir(exist_ok=True, parents=True)

    img_paths = find_block_image_paths(slab_path, ext=ext)
    processed_specimens = {p.stem for p in outpath.glob("*.png")}

    saved: List[Path] = []
    for img_path in img_paths:
        sp_name = img_path.stem.split("_")[0]
        is_block = len(sp_name.split(".")) == 6
        if not is_block and sp_name in processed_specimens:
            print(f"Skipping {sp_name} (already processed specimen)")
            continue
        print(f"Processing {sp_name}...")

        bgr, rgb = load_normalized_image(img_path)
        oct_mask = get_oct_mask(predictor, bgr, rgb, annotate=annotate_oct_mask)
        bgr_cropped, rgb_cropped, oct_cropped = crop_to_oct(bgr, oct_mask)

        tissue_mask, bbox = segment_tissue(predictor, rgb_cropped, bgr_cropped)
        if tissue_mask is None:
            print(f"  Tissue annotation cancelled for {sp_name}, skipping.")
            continue

        if show_qc:
            show_annotation_qc(rgb_cropped, oct_cropped, bbox)

        rgba = rgba_from_rgb_and_mask(rgb_cropped, tissue_mask)
        if show_qc:
            plt.imshow(rgba)
            plt.axis('off')
            plt.show()

        saved.append(save_rgba_image(rgba, outpath, sp_name))
    return saved

def process_slab_unregistered(
    slab_path: Path,
    predictor,
    annotate_oct_mask: bool = True,
    show_qc: bool = True,
    ext: str = ".tif",
) -> List[Path]:
    """Segment + crop + save every block specimen image in a slab directory.

    For each block image in *slab_path*:

    1. Load and exposure-normalize the image.
    2. Get an OCT mask (interactive SAM or auto).
    3. Crop the image and the OCT mask to the OCT bounding box.
    4. Interactively segment the tissue inside the cropped image with SAM.
    5. Save the RGBA tissue cutout to ``slab_path/processed/{specimen}.png``.

    Specimens whose output already exists are skipped (mirrors the original
    notebook's behaviour: items with more than 6 dot-separated name
    components are deduped against the processed folder; 6-component
    "block" names are always reprocessed).

    Parameters
    ----------
    slab_path
        Directory containing the slab's block images.
    predictor
        A ``segment_anything.SamPredictor`` instance.
    annotate_oct_mask
        If true, prompt the user for an OCT bbox via SAM. Otherwise use
        the automatic green-OCT segmenter.
    show_qc
        If true, display matplotlib QC figures for each specimen.
    ext
        Image extension to look for in *slab_path*.

    Returns
    -------
    list of Path
        Paths of the newly written PNGs (skipped specimens are excluded).
    """
    slab_path = Path(slab_path)
    outpath = slab_path / "processed"
    outpath.mkdir(exist_ok=True, parents=True)

    img_paths = find_block_image_paths(slab_path, ext=ext)
    processed_specimens = {p.stem for p in outpath.glob("*.png")}

    saved: List[Path] = []
    for img_path in img_paths:
        sp_name = img_path.stem.split("_")[0]
        is_block = len(sp_name.split(".")) == 6
        if not is_block and sp_name in processed_specimens:
            print(f"Skipping {sp_name} (already processed specimen)")
            continue
        print(f"Processing {sp_name}...")

        bgr, rgb = load_normalized_image(img_path)
        oct_mask = get_oct_mask(predictor, bgr, rgb, annotate=annotate_oct_mask)
        bgr_cropped, rgb_cropped, oct_cropped = crop_to_oct(bgr, oct_mask)

        tissue_mask, bbox = segment_tissue(predictor, rgb_cropped, bgr_cropped)
        if tissue_mask is None:
            print(f"  Tissue annotation cancelled for {sp_name}, skipping.")
            continue

        if show_qc:
            show_annotation_qc(rgb_cropped, oct_cropped, bbox)

        rgba = rgba_from_rgb_and_mask(rgb_cropped, tissue_mask)
        if show_qc:
            plt.imshow(rgba)
            plt.axis('off')
            plt.show()

        saved.append(save_rgba_image(rgba, outpath, sp_name))
    return saved

def segment_slab_registered(
    img_paths: List[Path],
    predictor,
    output_dir: Path,
    show_qc: bool = True,
) -> List[Path]:
    """Segment + save every block specimen image in a *registered* slab.

    Assumes *img_paths* are already mutually registered (same pixel grid,
    e.g. produced by :func:`register_translations_via_magic_wand`) and
    all belong to the **same block** across multiple sections.

    The user annotates a tissue bounding box **once** on the first image;
    that single bbox is then reused as the SAM prompt for every image in
    the list.

    Outputs are written as PNG to ``output_dir/{specimen}.png``.

    Parameters
    ----------
    img_paths
        Ordered list of registered block images (all from the same block).
        The first path is used as the reference for tissue annotation.
    predictor
        A ``segment_anything.SamPredictor`` instance.
    output_dir
        Directory where the RGBA cutouts are written. Created if missing.
    show_qc
        If true, display matplotlib QC figures for each specimen.

    Returns
    -------
    list of Path
        Paths of the newly written PNGs. Empty if the user cancelled
        annotation.
    """
    img_paths = [Path(p) for p in img_paths]
    if not img_paths:
        return []

    outpath = Path(output_dir)
    outpath.mkdir(exist_ok=True, parents=True)

    # Load + exposure-normalize every image up front so we can share the
    # interactively-chosen tissue bbox across all of them.
    loaded = [load_normalized_image(p) for p in img_paths]

    # Annotate the tissue bbox once on the first (reference) image.
    ref_bgr, _ = loaded[0]
    print(f"Annotating tissue bbox on {img_paths[0].stem.split('_')[0]}...")
    points = annotate_two_points(ref_bgr)
    if points is None:
        print("  Tissue annotation cancelled; aborting block.")
        return []
    tissue_bbox = (points[0][0], points[0][1], points[1][0], points[1][1])

    saved: List[Path] = []
    for img_path, (_, rgb) in zip(img_paths, loaded):
        sp_name = img_path.stem.split("_")[0]
        print(f"Processing {sp_name}...")

        tissue_mask = sam_mask_from_bbox(predictor, rgb, tissue_bbox)
        rgba = rgba_from_rgb_and_mask(rgb, tissue_mask)
        rgba_cropped = crop_image(rgba, tissue_bbox)
        if show_qc:
            plt.imshow(rgba_cropped)
            plt.axis('off')
            plt.show()

        saved.append(save_rgba_image(rgba_cropped, outpath, sp_name))
    return saved
