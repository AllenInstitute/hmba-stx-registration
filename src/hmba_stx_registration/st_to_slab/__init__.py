"""
Stage 2 – Register spatial-TX coordinates to slab images.

This module chains the ST-to-blockface affine (from ``st_to_block``) with
blockface-to-slab affines parsed from SVG layout files to produce a single
3×3 affine mapping microscope µm → slab mm.

**Outputs** (written to ``registration_results_{date}/``):

* ``{section}_coarse_transform_to_slab_mm_{date}.json`` – transform manifest
* ``{section}_registration_block_qc_{date}.png``        – block QC (copy)
* ``{section}_coarse_registration_slab_qc_{date}.png``   – slab QC plot
* ``{section}_run_manifest_{date}.json``                 – run manifest

Public API
----------
- ``get_bf_affines_from_svg``         – SVG → {bf_id: affine_to_slab}
- ``get_specimen_name_from_barcode``  – barcode → specimen name
- ``barcode_transform_to_slab``       – compute full transform chain
- ``plot_coarse_registration_slab_qc``– slab QC figure
- ``process_barcode``                 – end-to-end single-barcode pipeline
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from hmba_stx_registration import Specimen

from ..svg_utils import (
    aggregate_affine,
    get_scale_affine,
    load_svg_to_dict,
    parse_svg_transform,
)
from ..utils import (
    create_run_manifest_json,
    generate_random_colormap,
    sync_dir_to_s3,
    transform_coordinates,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SVG → blockface-to-slab affines
# ---------------------------------------------------------------------------

def get_bf_affines_from_svg(
    svg_path: Path,
    base_path: Path,
) -> Dict[str, np.ndarray]:
    """Extract blockface-to-slab affines from an SVG layout file.

    Each ``<image>`` element whose ``@href`` starts with ``"blocks"`` is
    treated as a blockface.  The returned affine maps *blockface source
    pixels* to SVG / slab pixel coordinates.

    Parameters
    ----------
    svg_path : Path
        Path to the SVG file.
    base_path : Path
        Root directory that ``@href`` paths are relative to so blockface
        image dimensions can be read.

    Returns
    -------
    dict
        ``{bf_id: 3×3 affine}``  (slab-pixel units).
    """
    svg_dict = load_svg_to_dict(svg_path)
    images = svg_dict["svg"]["image"]
    if not isinstance(images, list):
        images = [images]

    blockface_dicts = [d for d in images if d["@href"].startswith("blocks")]

    bf_affines: Dict[str, np.ndarray] = {}
    for bf_dict in blockface_dicts:
        bf_id = bf_dict["@id"]
        # Skip entries that are not real blockface images
        if len(bf_id.split(".")) < 8:
            continue

        bf_img = plt.imread(str(base_path / bf_dict["@href"]))
        scale_x = float(bf_dict["@width"]) / bf_img.shape[1]
        scale_y = float(bf_dict["@height"]) / bf_img.shape[0]

        try:
            transform = parse_svg_transform(bf_dict["@transform"])
        except (KeyError, ValueError):
            transform = np.eye(3)

        affine_to_slab = aggregate_affine(
            1, float(bf_dict["@x"]), float(bf_dict["@y"]),
            scale_x, scale_y, transform,
        )
        bf_affines[bf_id] = affine_to_slab

    return bf_affines


# ---------------------------------------------------------------------------
# Barcode → specimen name
# ---------------------------------------------------------------------------

def get_specimen_name_from_barcode(
    barcode: str,
    metadata_df: pd.DataFrame,
) -> Optional[str]:
    """Resolve a barcode to a specimen name (``set_name.section``).

    Parameters
    ----------
    barcode : str
        Barcode identifier.
    metadata_df : pd.DataFrame
        Metadata table with ``barcode``, ``section``, ``specimen_set_name``.

    Returns
    -------
    str or None
    """
    row = metadata_df[metadata_df["barcode"] == barcode]
    if len(row) != 1:
        logger.warning("Expected 1 row for barcode %s, got %d", barcode, len(row))
        return None
    return f"{row['specimen_set_name'].values[0]}.{row['section'].values[0]}"


# ---------------------------------------------------------------------------
# Transform chaining: sptx µm → slab mm
# ---------------------------------------------------------------------------

def barcode_transform_to_slab(
    barcode: str,
    metadata_df: pd.DataFrame,
    bf_affines: Dict[str, np.ndarray],
    transforms_path: Path,
    um_per_px: float = 20,
) -> Optional[List[dict]]:
    """Compute the full sptx-to-slab-mm affine(s) for a barcode.

    The transform chain is::

        scale(1/um_per_px) → st_to_bf → bf_to_slab → scale(um_per_px/1000)

    Handles single-piece and multi-subset (split-section) specimens.

    Parameters
    ----------
    barcode : str
        Barcode identifier.
    metadata_df : pd.DataFrame
        Metadata table.
    bf_affines : dict
        Pre-computed ``{bf_id: affine_to_slab}`` from :func:`get_bf_affines_from_svg`.
    transforms_path : Path
        Directory containing ``*_sptx_to_bf_affine.npy`` files.
    um_per_px : float
        Microns per pixel.

    Returns
    -------
    list[dict] or None
        Each dict contains ``source``, ``subset_label``, and ``transform``
        (3×3 list-of-lists).  Returns ``None`` if the ST-to-BF affine is
        missing.
    """

    def _create_transform_dict(bf_to_slab, st_to_bf, um_per_px, specimen_name, subset_label):
        scale_to_bf = get_scale_affine(1 / um_per_px, 1 / um_per_px)
        scale_to_mm = get_scale_affine(um_per_px / 1000, um_per_px / 1000)
        sptx_to_slab_mm = scale_to_mm @ bf_to_slab @ st_to_bf @ scale_to_bf
        return {
            "source": specimen_name,
            "subset_label": subset_label,
            "source_unit": "micrometer",
            "source_origin": "bottomleft",
            "target_unit": "millimeter",
            "target_origin": "topleft",
            "axis_order": "xy",
            "transform": sptx_to_slab_mm.tolist(),
        }

    def _find_matching_bf_key(specimen_name, specimen_set_name, section, bf_affines):
        if specimen_name in bf_affines:
            return specimen_name, True
        if f"{specimen_name}_0" in bf_affines:
            return specimen_name, True

        candidate_sections: Dict[str, int] = {}
        for key in bf_affines:
            base_key = key.split("_")[0]
            parts = base_key.split(".")
            if len(parts) >= 8:
                key_set = ".".join(parts[:7])
                key_section = parts[7]
                if key_set == specimen_set_name:
                    candidate_sections[base_key] = int(key_section)

        if not candidate_sections:
            return None, False

        try:
            target_section = int(section)
        except (ValueError, TypeError):
            return None, False

        closest_key = min(
            candidate_sections, key=lambda k: abs(candidate_sections[k] - target_section),
        )
        closest_section = candidate_sections[closest_key]
        logger.warning(
            "No exact blockface match for %s. Using closest section: %s "
            "(section %s vs %s)",
            specimen_name, closest_key, closest_section, target_section,
        )
        return closest_key, False

    # --- resolve barcode ---
    row = metadata_df[metadata_df["barcode"] == barcode]
    assert len(row) == 1, f"Expected 1 row for barcode {barcode}, got {len(row)}"
    specimen_set_name = row["specimen_set_name"].values[0]
    section = row["section"].values[0]
    specimen_name = f"{specimen_set_name}.{section}"

    # --- load ST→BF affine ---
    try:
        st_to_bf = np.load(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy")
    except FileNotFoundError:
        logger.info("No affine transform found for %s, skipping", specimen_name)
        return None

    matched_key, _ = _find_matching_bf_key(
        specimen_name, specimen_set_name, section, bf_affines,
    )
    if matched_key is None:
        logger.warning("No blockface affine for %s in bf_affines, skipping", specimen_name)
        return None

    # --- build transform manifest ---
    transform_manifest: List[dict] = []
    if matched_key in bf_affines:
        # single piece
        transform_manifest.append(
            _create_transform_dict(bf_affines[matched_key], st_to_bf, um_per_px, specimen_name, "nan"),
        )
    else:
        # multi-subset
        subset_label = 0
        while True:
            key = f"{matched_key}_{subset_label}"
            bf_to_slab = bf_affines.get(key)
            if bf_to_slab is None:
                break
            transform_manifest.append(
                _create_transform_dict(bf_to_slab, st_to_bf, um_per_px, specimen_name, subset_label),
            )
            subset_label += 1
        if not transform_manifest:
            logger.warning("No blockface affine for %s in bf_affines, skipping", matched_key)
            return None

    return transform_manifest


# ---------------------------------------------------------------------------
# QC plotting
# ---------------------------------------------------------------------------

def plot_coarse_registration_slab_qc(
    slab_img: np.ndarray,
    coords_mm: np.ndarray,
    label_color: pd.Series,
    mm_per_px: float,
    output_path: Path,
) -> None:
    """Save a QC plot of transformed coordinates overlaid on a slab image.

    Parameters
    ----------
    slab_img : np.ndarray
        Slab frozen image.
    coords_mm : np.ndarray, shape (N, 2+)
        Transformed coordinates in mm (x, y, …).
    label_color : pd.Series
        Per-cell colour values (hex strings or similar).
    mm_per_px : float
        Scale factor (mm per pixel) of *slab_img*.
    output_path : Path
        Destination path for the QC PNG.
    """
    H, W = slab_img.shape[:2]
    x_max = W * mm_per_px
    y_max = H * mm_per_px

    plt.figure(figsize=(10, 10))
    plt.imshow(slab_img, extent=[0, x_max, y_max, 0], origin="upper")
    plt.scatter(
        coords_mm[::50, 0], coords_mm[::50, 1],
        c=label_color[::50], alpha=0.3, s=1,
    )
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title(output_path.stem)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# End-to-end single-barcode pipeline
# ---------------------------------------------------------------------------

def process_barcode(
    specimen: Specimen,
    bf_affines: Dict[str, np.ndarray],
    slab_imgs: Dict[int, np.ndarray],
    transforms_path: Path,
    um_per_px: float,
    date: str,
    version: str,
    sync_to_s3: bool = False,
    sync_dryrun: bool = True,
) -> Optional[dict]:
    """Process one barcode end-to-end, producing all Stage-2 outputs.

    Outputs written to ``{base_barcodes_path}/{barcode}/registration_results_{date}/``:

    1. ``{section}_coarse_transform_to_slab_mm_{date}.json``
    2. ``{section}_registration_block_qc_{date}.png``
    3. ``{section}_coarse_registration_slab_qc_{date}.png``  (per subset)
    4. ``{section}_run_manifest_{date}.json``

    Parameters
    ----------
    specimen : Specimen
        Specimen object encapsulating the barcode, its metadata, and the
        base barcodes path.
    bf_affines : dict
        Blockface-to-slab affines (pixel units).
    slab_imgs : dict
        Slab images keyed by integer slab id.
    transforms_path : Path
        Directory with ``*_sptx_to_bf_affine.npy`` files and ``qc/`` subfolder.
    um_per_px : float
        Microns per pixel.
    date : str
        Date string (``yyyymmdd``) for output filenames.
    version : str
        Version string stored in the run manifest (e.g. ``"0.1.0"``).
    sync_to_s3 : bool
        If ``True``, sync results to S3 after processing.
    sync_dryrun : bool
        If ``True``, S3 sync is a dry run only.

    Returns
    -------
    dict or None
        Summary with ``barcode``, ``specimen_name``, ``transform_manifest``,
        ``output_files``.  ``None`` on early skip.
    """
    barcode = specimen.barcode
    specimen_name = specimen.specimen_name
    barcode_path = specimen.barcode_path

    mm_per_px = um_per_px / 1000
    results_path = barcode_path / f"registration_results_{date}"
    input_files: List[Path] = []
    output_files: List[Path] = []

    # Build a single-row metadata DataFrame for barcode_transform_to_slab
    metadata_row = pd.DataFrame([{
        "barcode": barcode,
        "specimen_set_name": specimen.specimen_set_name,
        "section": specimen.metadata["section"],
    }])

    # 1. Compute transform chain
    try:
        transform_meta = barcode_transform_to_slab(
            barcode, metadata_row, bf_affines, transforms_path, um_per_px,
        )
        if transform_meta is None:
            logger.info("[%s] No transforms computed, skipping", barcode)
            return None
    except Exception:
        logger.exception("[%s] Failed to compute transforms", barcode)
        return None

    # 2. Load cell table
    try:
        slab_id = int(specimen.metadata["slab"])
        table_path = next(barcode_path.glob(f"{specimen_name}_mapping_for_registration_*.csv"))
        input_files.append(table_path)

        try:
            col_names_path = next(
                barcode_path.glob(f"{specimen_name}_column_names_for_registration_*.json"),
            )
            col_names = json.load(open(col_names_path, 'r'))
            table_label = col_names[0]
            input_files.append(col_names_path)
        except StopIteration:
            logger.info("[%s] No column_names_for_registration JSON found", barcode)

        table = pd.read_csv(table_path, index_col=0)
        label_cmap = generate_random_colormap(table[table_label].unique(), seed=44)
        label_color = table[table_label].map(label_cmap)
    except Exception:
        logger.exception("[%s] Failed to load cell table", barcode)
        return None

    results_path.mkdir(parents=True, exist_ok=True)

    # 3. Copy block QC
    try:
        st2block_qc_src = transforms_path / "qc" / f"{specimen_name}_registration_qc.png"
        if st2block_qc_src.exists():
            st2block_qc_dst = results_path / f"{specimen_name}_registration_block_qc_{date}.png"
            shutil.copy(st2block_qc_src, st2block_qc_dst)
            output_files.append(st2block_qc_dst)
            logger.info("[%s] Copied block QC → %s", barcode, st2block_qc_dst.name)
        else:
            logger.info("[%s] No st2block QC at %s", barcode, st2block_qc_src)
    except Exception:
        logger.exception("[%s] Failed to copy st2block QC", barcode)

    # 4. Slab QC per subset
    for meta in transform_meta:
        try:
            coords = table[["center_x", "center_y"]]
            subset_label = meta["subset_label"]

            if subset_label != "nan":
                subset_table_path = next(barcode_path.glob("*subset_label*.csv"))
                subset_table = pd.read_csv(subset_table_path, index_col=0)
                mask = subset_table["subset_label"] == subset_label
                coords = coords[mask]
                cur_label_color = label_color[mask]
            else:
                cur_label_color = label_color

            tfm = np.array(meta["transform"])
            transformed_mm = transform_coordinates(coords.values, tfm)

            suffix = f"_{subset_label}" if subset_label != "nan" else ""
            qc_filename = f"{specimen_name}{suffix}_coarse_registration_slab_qc_{date}.png"
            output_files.append(qc_filename)
            qc_out = results_path / qc_filename
            plot_coarse_registration_slab_qc(
                slab_imgs[slab_id], transformed_mm, cur_label_color, mm_per_px, qc_out,
            )
            output_files.append(qc_out)
        except Exception:
            logger.exception(
                "[%s] Failed slab QC for subset %s", barcode, meta.get("subset_label"),
            )

    # 5. Save transform manifest JSON
    try:
        manifest_path = results_path / f"{specimen_name}_coarse_transform_to_slab_mm_{date}.json"
        with open(manifest_path, "w") as fh:
            json.dump(transform_meta, fh, indent=4)
        output_files.append(manifest_path)
        logger.info("[%s] Saved transform manifest: %s", barcode, manifest_path.name)
    except Exception:
        logger.exception("[%s] Failed to save transform manifest", barcode)

    # 6. Run manifest
    try:
        run_manifest_path = results_path / f"{specimen_name}_run_manifest_{date}.json"
        create_run_manifest_json(
            ver=version,
            date=date,
            specimen_name=specimen_name,
            input_files=input_files,
            output_files=output_files,
            args={"um_per_px": um_per_px, "table_label": table_label},
            out_path=run_manifest_path,
        )
        logger.info("[%s] Saved run manifest: %s", barcode, run_manifest_path.name)
    except Exception:
        logger.exception("[%s] Failed to save run manifest", barcode)

    # 7. Optional S3 sync
    if sync_to_s3:
        try:
            sync_dir_to_s3(
                local_dir=results_path,
                bucket="hmba-macaque-wg-802451596237-us-west-2",
                s3_dir=f"hmba_aim_2/Xenium/QM24.50.002/{barcode}/registration_results_{date}",
                profile="storage",
                dryrun=sync_dryrun,
                delete=False,
            )
            logger.info("[%s] Synced to S3", barcode)
        except Exception:
            logger.exception("[%s] S3 sync failed", barcode)

    logger.info("[%s] Done (%s)", barcode, specimen_name)
    return {
        "barcode": barcode,
        "specimen_name": specimen_name,
        "transform_manifest": transform_meta,
        "output_files": [str(p) for p in output_files],
    }
