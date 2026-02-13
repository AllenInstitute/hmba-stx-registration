"""
Shared utilities for the HMBA spatial-TX registration pipeline.

This module provides:
- ``generate_random_colormap`` – categorical label → colour mapping for QC plots
- ``transform_coordinates``    – apply a 3×3 affine to 2-D points
- ``cluster_coordinates``      – DBSCAN-based split-section clustering
- ``create_run_manifest_json`` – write a run-manifest JSON file
- ``sync_dir_to_s3``           – thin wrapper around ``aws s3 sync``
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colourmap helper (reimplemented – original source no longer available)
# ---------------------------------------------------------------------------

def generate_random_colormap(
    labels: Sequence[str],
    seed: int = 42,
) -> Dict[str, str]:
    """Create a reproducible mapping from categorical *labels* to hex colours.

    Parameters
    ----------
    labels : sequence of str
        Unique label values.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{label: "#RRGGBB"}`` mapping.
    """
    rng = np.random.RandomState(seed)
    cmap: Dict[str, str] = {}
    for label in labels:
        r, g, b = rng.randint(0, 256, size=3)
        cmap[label] = f"#{r:02x}{g:02x}{b:02x}"
    return cmap


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def transform_coordinates(
    coordinates: np.ndarray,
    transform_matrix: np.ndarray,
) -> np.ndarray:
    """Apply a 3×3 affine *transform_matrix* to 2-D *coordinates*.

    Parameters
    ----------
    coordinates : np.ndarray, shape (N, 2)
        Input coordinates (x, y).
    transform_matrix : np.ndarray, shape (3, 3)
        Homogeneous affine matrix.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Transformed coordinates (the third column is always 1).

    Notes
    -----
    Input coordinates are in microscope µm (origin bottom-left, x/y order).
    Output coordinates are in slab mm (origin top-left, x/y order) when
    using the full pipeline transform.
    """
    coords = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    return (transform_matrix @ coords.T).T


# ---------------------------------------------------------------------------
# Split-section clustering
# ---------------------------------------------------------------------------

def cluster_coordinates(
    table: pd.DataFrame,
    coords_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Cluster cells into two spatial groups using DBSCAN.

    Used when a tissue section is physically split into two chunks that each
    require an independent affine transform to the slab.

    The function adds a ``subset_label`` column to the returned table:
    * 0 → top cluster (lower *y*)
    * 1 → bottom cluster (higher *y*)

    Parameters
    ----------
    table : pd.DataFrame
        Cell table with at least the columns listed in *coords_cols*.
    coords_cols : list of str, optional
        Column names for the 2-D coordinates used for clustering.
        Defaults to ``["x", "y_blockface_px"]``.

    Returns
    -------
    pd.DataFrame
        A copy of *table* with a new ``subset_label`` column.
    """
    from sklearn.cluster import DBSCAN  # local import to keep dep optional

    if coords_cols is None:
        coords_cols = ["x", "y_blockface_px"]

    table = table.copy()
    coords = table[coords_cols].values

    dbscan = DBSCAN(eps=500, min_samples=10)
    cluster_labels = dbscan.fit_predict(coords)

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    logger.info("DBSCAN clusters: %s", dict(zip(unique_labels.tolist(), counts.tolist())))

    # Keep only the two largest clusters; merge everything else
    if len(unique_labels) > 2:
        label_counts = sorted(
            [(l, c) for l, c in zip(unique_labels, counts) if l != -1],
            key=lambda x: x[1],
            reverse=True,
        )
        top_2 = [label_counts[0][0], label_counts[1][0]]
        for old_label in unique_labels:
            if old_label not in top_2:
                mask = cluster_labels == old_label
                pts = coords[mask]
                if len(pts) > 0:
                    centre = pts.mean(axis=0)
                    d0 = np.linalg.norm(centre - coords[cluster_labels == top_2[0]].mean(axis=0))
                    d1 = np.linalg.norm(centre - coords[cluster_labels == top_2[1]].mean(axis=0))
                    cluster_labels[mask] = top_2[0] if d0 < d1 else top_2[1]
        cluster_labels = (cluster_labels == top_2[1]).astype(int)

    # Ensure cluster 0 = top (lower y), cluster 1 = bottom (higher y)
    if coords[cluster_labels == 0, 1].mean() > coords[cluster_labels == 1, 1].mean():
        cluster_labels = 1 - cluster_labels

    table["subset_label"] = cluster_labels
    logger.info(
        "Cluster sizes: 0=%d, 1=%d",
        (cluster_labels == 0).sum(),
        (cluster_labels == 1).sum(),
    )
    return table


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

# Current manifest schema version – bump when the schema changes.
_MANIFEST_SCHEMA_VERSION = "1.0"


def create_run_manifest_json(
    ver: str,
    date: str,
    specimen_name: str,
    input_files: List[Union[str, Path]],
    output_files: List[Union[str, Path]],
    args: Optional[Dict] = None,
    out_path: Union[str, Path] = "run_manifest.json",
) -> None:
    """Write a run-manifest JSON file.

    Output filename convention::

        {section_name}_run_manifest_{yyyymmdd}.json

    Parameters
    ----------
    ver : str
        Version of ``hmba_stx_registration`` (e.g. ``"0.1.0"``).
    date : str
        Run date as ``yyyymmdd``.
    specimen_name : str
        Section identifier (e.g. ``"QM24.50.002.CX.51.01.05.02"``).
    input_files, output_files : list of str/Path
        Files consumed / produced by the run.
    args : dict, optional
        Parameters used in the run.
    out_path : str or Path
        Destination path for the manifest JSON.
    """
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "hmba_stx_registration_version": str(ver),
        "date": date,
        "specimen_name": specimen_name,
        "input_files": [str(Path(f).name) for f in input_files],
        "output_files": [str(Path(f).name) for f in output_files],
        "args": args if args is not None else {},
    }
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=4)


# ---------------------------------------------------------------------------
# S3 sync
# ---------------------------------------------------------------------------

def sync_dir_to_s3(
    local_dir: Union[str, Path],
    bucket: str,
    s3_dir: str,
    profile: str,
    dryrun: bool = False,
    delete: bool = False,
) -> None:
    """Sync a local directory to S3 using ``aws s3 sync``.

    Parameters
    ----------
    local_dir : str or Path
        Local directory to upload.
    bucket : str
        S3 bucket name.
    s3_dir : str
        Key prefix in the bucket.
    profile : str
        AWS CLI profile name.
    dryrun : bool
        Pass ``--dryrun`` to aws-cli.
    delete : bool
        Pass ``--delete`` to aws-cli (remove remote files not present locally).
    """
    s3_target = f"s3://{bucket.rstrip('/')}/{s3_dir.lstrip('/')}"
    cmd = (
        f"aws s3 sync {shlex.quote(str(local_dir))} "
        f"{shlex.quote(s3_target)} "
        f"--profile {shlex.quote(profile)} "
        "--only-show-errors"
    )
    if dryrun:
        cmd += " --dryrun"
    if delete:
        cmd += " --delete"
    logger.info("Running: %s", cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("aws s3 sync failed (exit %d)", exc.returncode)
        raise
