"""
HMBA Spatial TX Registration
=============================

Tools for registering HMBA spatial transcriptomics data to Allen Brain Atlas
coordinate systems.

The pipeline has two stages:

Stage 1 – :mod:`hmba_stx_registration.st_to_block`
    Register STx cell coordinates to blockface images.  Produces a 3×3 affine
    (sptx pixel → blockface pixel) and a 6-panel QC image.

Stage 2 – :mod:`hmba_stx_registration.st_to_slab`
    Chain the ST→block affine with blockface→slab affines parsed from SVG
    layout files to produce a single 3×3 affine (microscope µm → slab mm).
    Writes transform manifests, QC images, and run manifests.

Shared helpers live in :mod:`hmba_stx_registration.utils` and
:mod:`hmba_stx_registration.svg_utils`.
"""

from functools import cached_property
from .st_to_block import register_cells_to_blockface
from .st_to_slab import barcode_transform_to_slab, process_barcode
from .utils import transform_coordinates, upload_plots_to_s3
import logging
import numpy as np
import pandas as pd
import json
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    "register_cells_to_blockface",
    "barcode_transform_to_slab",
    "process_barcode",
    "get_metadata_df",
    "apply_slab_transforms",
    "plot_slab_scatter",
    "plot_single_slab_scatter",
    "upload_plots_to_s3",
    "Specimen",
]

def get_metadata_df(barcodes_path: Union[str, Path]) -> pd.DataFrame:
    if isinstance(barcodes_path, str):
        barcodes_path = Path(barcodes_path)
    metadata = []
    for path in barcodes_path.glob('*'):
        if not path.is_dir():
            print(path, "is not a directory, skipping")
            continue
        # read json
        json_path = path / f"{path.name}_specimen_metadata.json"
        if json_path.exists() is False:
            print(json_path, "does not exist, skipping")
            continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        metadata.append(data)
    metadata = pd.DataFrame(metadata)
    metadata['specimen_set_name'] = metadata['donor'] + '.' + metadata['division'] + '.' + metadata['slab'] + '.' + metadata['block'] + '.' + metadata['set']
    metadata['specimen_name'] = metadata['donor'] + '.' + metadata['division'] + '.' + metadata['slab'] + '.' + metadata['block'] + '.' + metadata['set'] + '.' + metadata['section']
    return metadata

class Specimen:
    def __init__(self, barcode: str, base_barcodes_path: Union[str, Path], date: Optional[str] = None):
        self.barcode = barcode
        self.base_barcodes_path = Path(base_barcodes_path)
        self.barcode_path = self.base_barcodes_path / self.barcode
        json_path = self.barcode_path / f"{self.barcode}_specimen_metadata.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.metadata = data
        self.slab_name = data['donor'] + '.' + data['division'] + '.' + data['slab']
        self.specimen_set_name = data['donor'] + '.' + data['division'] + '.' + data['slab'] + '.' + data['block'] + '.' + data['set']
        self.specimen_name = data['donor'] + '.' + data['division'] + '.' + data['slab'] + '.' + data['block'] + '.' + data['set'] + '.' + data['section']
        self.date = date

    @cached_property
    def cells_table(self) -> pd.DataFrame:
        cells_path = Path('')
        if self.date:
            cells_path = self.barcode_path / f"{self.specimen_name}_mapping_for_registration_{self.date}.csv"
        if not cells_path.exists() or self.date is None:
            cells_path = next(self.barcode_path.glob(f"{self.specimen_name}_mapping_for_registration*.csv"))
        if not cells_path.exists():
            raise FileNotFoundError(f"Cells file not found: {cells_path}")
        cells_table = pd.read_csv(cells_path, index_col=0)
        if {"x", "y"}.issubset(cells_table.columns) and not {"center_x", "center_y"}.issubset(cells_table.columns):
            cells_table = cells_table.rename(columns={"x": "center_x", "y": "center_y"})
        # check if center_x and center_y contain NaNs, throw warning and filter them out and continue
        if cells_table["center_x"].isna().any() or cells_table["center_y"].isna().any():
            n_dropped = int(cells_table["center_x"].isna().sum() + cells_table["center_y"].isna().sum())
            logger.warning(
                f"Cells table contains {n_dropped} rows with NaN center_x or center_y. These will be dropped for downstream processing.",
            )
            cells_table = cells_table.dropna(subset=["center_x", "center_y"])
        return cells_table

    @cached_property
    def table_label(self) -> str:
        mapping_columns_path = Path('')
        if self.date:
            mapping_columns_path = self.barcode_path / f"{self.specimen_name}_column_names_for_registration_{self.date}.json"
        if not mapping_columns_path.exists() or self.date is None:
            mapping_columns_path = next(self.barcode_path.glob(f"{self.specimen_name}_column_names_for_registration*.json"))
        if not mapping_columns_path.exists():
            raise FileNotFoundError(f"Mapping columns file not found: {mapping_columns_path}")
        with open(mapping_columns_path, 'r') as f:
            columns = eval(f.read())
        return columns[0]


def apply_slab_transforms(
    barcode_path: Union[str, Path],
    registration_results_date: str,
    barcodes: Optional[list] = None,
    mapping_date: Optional[str] = None,
) -> pd.DataFrame:
    """Apply slab transforms to cell tables across all barcode directories.

    For each barcode directory under *barcode_path*, loads the cells table
    via :class:`Specimen` and applies the sptx-to-slab-mm affine stored in::

        {barcode_dir}/registration_results_{date}/
            {specimen_name}_coarse_transform_to_slab_mm_{date}.json

    Two columns are appended to each table:

    * ``x_slab_mm`` – transformed x coordinate in slab mm
    * ``y_slab_mm`` – transformed y coordinate in slab mm

    All per-barcode tables are concatenated into a single master DataFrame.

    Parameters
    ----------
    barcode_path : str or Path
        Root directory containing one sub-directory per barcode.
    registration_results_date : str
        Date string (``YYYYMMDD``) identifying which
        ``registration_results_{date}`` directory to read transforms from.
    barcodes : list, optional
        If given, only process these barcode directory names.  When *None*
        (the default), every numerically named sub-directory of
        *barcode_path* is processed.
    mapping_date : str, optional
        Date string (``YYYYMMDD``) passed to :class:`Specimen` to select a
        specific ``*_mapping_for_registration_{date}.csv``.  If *None*,
        :class:`Specimen` picks whichever matching CSV it finds.

    Returns
    -------
    pd.DataFrame
        Master table with all barcodes, including ``barcode`` and
        ``specimen_name`` columns plus ``x_slab_mm`` / ``y_slab_mm``.
    """
    barcode_path = Path(barcode_path)
    results_dir_name = f"registration_results_{registration_results_date}"

    all_dfs = []

    if barcodes is not None:
        bc_names = [str(b) for b in barcodes]
    else:
        bc_names = sorted(
            d.name for d in barcode_path.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

    for barcode in bc_names:
        bc_dir = barcode_path / barcode
        if not bc_dir.is_dir():
            continue

        # -- load specimen metadata + cells table via Specimen --
        try:
            specimen = Specimen(barcode, barcode_path, date=mapping_date)
        except FileNotFoundError:
            logger.warning("[%s] No metadata JSON found, skipping", barcode)
            continue

        specimen_name = specimen.specimen_name

        # -- load the transform manifest --
        transform_path = (
            bc_dir
            / results_dir_name
            / f"{specimen_name}_coarse_transform_to_slab_mm_{registration_results_date}.json"
        )
        if not transform_path.exists():
            logger.warning(
                "[%s] Transform manifest not found at %s, skipping", barcode, transform_path
            )
            continue
        with open(transform_path) as fh:
            transform_meta = json.load(fh)

        # -- read cell table and tag with identifiers --
        try:
            table = specimen.cells_table.copy()
        except (FileNotFoundError, StopIteration):
            logger.warning("[%s] No mapping CSV found, skipping", barcode)
            continue
        if not {"center_x", "center_y"}.issubset(table.columns):
            logger.warning(
                "[%s] No center_x/center_y columns found in cell table, skipping",
                barcode,
            )
            continue
        table["barcode"] = barcode
        table["specimen_name"] = specimen_name

        # -- apply transform(s) --
        is_split = any(
            str(m.get("subset_label")) != "nan" for m in transform_meta
        )

        if not is_split:
            tfm = np.array(transform_meta[0]["transform"])
            coords = table[["center_x", "center_y"]].values
            transformed = transform_coordinates(coords, tfm)
            table["x_slab_mm"] = transformed[:, 0]
            table["y_slab_mm"] = transformed[:, 1]
        else:
            # Split section: each subset has its own affine
            subset_candidates = list(bc_dir.glob("*subset_label*.csv"))
            if not subset_candidates:
                logger.warning(
                    "[%s] Split section but no subset_label CSV found, skipping", barcode
                )
                continue
            subset_table = pd.read_csv(subset_candidates[0], index_col=0)
            table["x_slab_mm"] = np.nan
            table["y_slab_mm"] = np.nan
            for meta_entry in transform_meta:
                sl = meta_entry["subset_label"]
                mask = subset_table["subset_label"] == sl
                if not mask.any():
                    continue
                tfm = np.array(meta_entry["transform"])
                coords = table.loc[mask, ["center_x", "center_y"]].values
                transformed = transform_coordinates(coords, tfm)
                table.loc[mask, "x_slab_mm"] = transformed[:, 0]
                table.loc[mask, "y_slab_mm"] = transformed[:, 1]

        all_dfs.append(table)
        logger.info("[%s] Appended %d rows", barcode, len(table))

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def plot_slab_scatter(
    df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
    label_col: str = "supercluster_term_label",
    slab_col: str = "slab",
    plane_col: str = "slab_plane",
    join_col: str = "barcode",
    exclude_labels: Optional[list] = None,
    seed: int = 40,
    point_size: float = 1.0,
    alpha: float = 0.5,
) -> Dict[tuple, plt.Figure]:
    """Create scatter plots of cell coordinates in slab space.

    Produces one :class:`matplotlib.figure.Figure` per unique
    ``(slab, slab_plane)`` combination, with cells plotted at their
    ``x_slab_mm`` / ``y_slab_mm`` positions and coloured by *label_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Master table returned by :func:`apply_slab_transforms`, which must
        contain ``x_slab_mm``, ``y_slab_mm``, and *label_col* columns.
    metadata_df : pd.DataFrame, optional
        External metadata table (e.g. the section-metadata CSV) that
        supplies *slab_col* and/or *plane_col* when those columns are absent
        from *df*.  Joined to *df* on *join_col*.
    label_col : str
        Column to use for colour categories.
        Defaults to ``"supercluster_term_label"``.
    slab_col : str
        Column identifying the slab.  Defaults to ``"slab"``.
    plane_col : str
        Column identifying the plane within a slab.  Defaults to
        ``"slab_plane"``.
    join_col : str
        Column used to join *metadata_df* onto *df*.  Defaults to
        ``"barcode"``.
    seed : int
        Random seed passed to :func:`~hmba_stx_registration.utils.generate_random_colormap`.
    point_size : float
        Marker size for scatter points.
    alpha : float
        Marker transparency.

    Returns
    -------
    dict
        ``{(slab, slab_plane): Figure}`` mapping so callers can save or
        show individual figures.
    """
    from .utils import generate_random_colormap

    # Drop columns not used
    df = df.copy()
    needed_cols = {"x_slab_mm", "y_slab_mm", label_col, slab_col}
    if plane_col in df.columns:
        needed_cols.add(plane_col)
    df = df[list(needed_cols.intersection(df.columns)) + [join_col]]

    # Merge metadata columns that are missing from df
    if metadata_df is not None:
        needed = [c for c in [slab_col, plane_col] if c not in df.columns]
        if needed and join_col in metadata_df.columns:
            cols_to_merge = [join_col] + [c for c in needed if c in metadata_df.columns]
            # Align dtypes on the join key to avoid merge errors
            df = df.copy()
            df[join_col] = df[join_col].astype(str)
            right = metadata_df[cols_to_merge].copy()
            right[join_col] = right[join_col].astype(str)
            df = df.merge(right, on=join_col, how="left")

    required = {"x_slab_mm", "y_slab_mm", slab_col, plane_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # Build a colour map over all unique labels in the full dataset
    all_labels = df[label_col].dropna().unique()
    cmap = generate_random_colormap(sorted(all_labels), seed=seed)

    exclude_set = set(exclude_labels) if exclude_labels else set()

    groups = _group_by_slab_plane(df, slab_col, plane_col)

    figures: Dict[tuple, plt.Figure] = {}
    for (slab, plane), group in groups.items():
        fig = plot_single_slab_scatter(
            group,
            label_col=label_col,
            cmap=cmap,
            exclude_labels=exclude_set,
            title=f"Slab {slab} – plane {plane}",
            point_size=point_size,
            alpha=alpha,
        )
        figures[(slab, plane)] = fig

    return figures


def _group_by_slab_plane(
    df: pd.DataFrame,
    slab_col: str,
    plane_col: str,
) -> Dict[tuple, pd.DataFrame]:
    """Split *df* into sub-DataFrames keyed by ``(slab, plane)``.

    Returns a dict so callers can inspect exactly which groups exist and
    iterate at their own pace.
    """
    return {
        key: group
        for key, group in df.groupby([slab_col, plane_col], sort=True)
    }


def plot_single_slab_scatter(
    df: pd.DataFrame,
    label_col: str = "supercluster_term_label",
    cmap: Optional[Dict[str, str]] = None,
    exclude_labels: Optional[set] = None,
    title: str = "",
    point_size: float = 1.0,
    alpha: float = 0.5,
    seed: int = 40,
) -> plt.Figure:
    """Plot a single slab-plane scatter from an already-subsetted DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``x_slab_mm``, ``y_slab_mm``, and *label_col*.
    label_col : str
        Column used for colour categories.
    cmap : dict, optional
        ``{label: colour}`` mapping.  Built automatically from the data
        when *None*.
    exclude_labels : set, optional
        Labels to exclude from the plot.
    title : str
        Axes title.
    point_size : float
        Marker size.
    alpha : float
        Marker transparency.
    seed : int
        Random seed for colour-map generation when *cmap* is not supplied.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .utils import generate_random_colormap

    valid = df.dropna(subset=["x_slab_mm", "y_slab_mm"])
    if exclude_labels:
        valid = valid[~valid[label_col].isin(exclude_labels)]

    if cmap is None:
        all_labels = valid[label_col].dropna().unique()
        cmap = generate_random_colormap(sorted(all_labels), seed=seed)

    fig, ax = plt.subplots(figsize=(10, 8))

    for label, label_group in valid.groupby(label_col, sort=True):
        color = cmap.get(str(label), "#888888")
        ax.scatter(
            label_group["x_slab_mm"],
            label_group["y_slab_mm"],
            c=color,
            s=point_size,
            alpha=alpha,
            label=str(label),
            rasterized=True,
        )

    ax.set_xlabel("x slab (mm)")
    ax.set_ylabel("y slab (mm)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # slab origin is top-left
    ax.legend(
        markerscale=6,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=7,
    )
    fig.tight_layout()

    return fig

