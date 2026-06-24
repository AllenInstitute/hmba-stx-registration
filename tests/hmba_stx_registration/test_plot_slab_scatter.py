"""Tests for plot_slab_scatter, plot_single_slab_scatter, and _group_by_slab_plane."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from hmba_stx_registration import (
    _group_by_slab_plane,
    plot_single_slab_scatter,
    plot_slab_scatter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cells(
    n: int,
    slab: str,
    plane: str,
    block: str,
    barcode: str,
    label: str = "TypeA",
    rng: np.random.RandomState | None = None,
) -> pd.DataFrame:
    """Return a small cell table with slab coords already filled in."""
    if rng is None:
        rng = np.random.RandomState(0)
    return pd.DataFrame({
        "x_slab_mm": rng.uniform(0, 10, n),
        "y_slab_mm": rng.uniform(0, 10, n),
        "slab": slab,
        "slab_plane": plane,
        "block": block,
        "barcode": barcode,
        "supercluster_term_label": label,
    })


SLAB = "QM24.50.002.CX.49"
PLANE = "Plane01"


def _make_multi_block_df() -> pd.DataFrame:
    """9 blocks under the same slab/plane, 50 cells each => 450 rows."""
    rng = np.random.RandomState(42)
    blocks = []
    for i in range(1, 10):
        blocks.append(
            _make_cells(
                50,
                slab=SLAB,
                plane=PLANE,
                block=f"B{i}",
                barcode=f"BC{i:04d}",
                label=rng.choice(["TypeA", "TypeB", "TypeC"]),
                rng=rng,
            )
        )
    return pd.concat(blocks, ignore_index=True)


# ---------------------------------------------------------------------------
# _group_by_slab_plane
# ---------------------------------------------------------------------------

class TestGroupBySlabPlane:
    def test_single_group(self):
        df = _make_multi_block_df()
        groups = _group_by_slab_plane(df, "slab", "slab_plane")
        assert len(groups) == 1
        key = (SLAB, PLANE)
        assert key in groups
        assert len(groups[key]) == 450

    def test_multiple_groups(self):
        df1 = _make_cells(30, "SlabA", "P1", "B1", "bc1")
        df2 = _make_cells(20, "SlabA", "P2", "B2", "bc2")
        df3 = _make_cells(10, "SlabB", "P1", "B3", "bc3")
        df = pd.concat([df1, df2, df3], ignore_index=True)
        groups = _group_by_slab_plane(df, "slab", "slab_plane")
        assert len(groups) == 3
        assert len(groups[("SlabA", "P1")]) == 30
        assert len(groups[("SlabA", "P2")]) == 20
        assert len(groups[("SlabB", "P1")]) == 10

    def test_nan_slab_drops_rows(self):
        """groupby silently drops NaN keys — verify this is what happens."""
        df = _make_cells(20, SLAB, PLANE, "B1", "bc1")
        df_nan = _make_cells(10, SLAB, PLANE, "B2", "bc2")
        df_nan["slab"] = np.nan  # simulate missing metadata
        combined = pd.concat([df, df_nan], ignore_index=True)
        groups = _group_by_slab_plane(combined, "slab", "slab_plane")
        total = sum(len(g) for g in groups.values())
        # 10 rows with NaN slab are silently lost
        assert total == 20, (
            f"Expected 20 rows (NaN slab rows dropped), got {total}"
        )

    def test_nan_plane_drops_rows(self):
        """Same problem when plane is NaN."""
        df = _make_cells(20, SLAB, PLANE, "B1", "bc1")
        df_nan = _make_cells(10, SLAB, PLANE, "B2", "bc2")
        df_nan["slab_plane"] = np.nan
        combined = pd.concat([df, df_nan], ignore_index=True)
        groups = _group_by_slab_plane(combined, "slab", "slab_plane")
        total = sum(len(g) for g in groups.values())
        assert total == 20


# ---------------------------------------------------------------------------
# plot_single_slab_scatter
# ---------------------------------------------------------------------------

class TestPlotSingleSlabScatter:
    def test_returns_figure(self):
        df = _make_multi_block_df()
        fig = plot_single_slab_scatter(df, title="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_auto_cmap(self):
        df = _make_cells(30, SLAB, PLANE, "B1", "bc1", label="TypeX")
        fig = plot_single_slab_scatter(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_supplied_cmap(self):
        df = _make_cells(30, SLAB, PLANE, "B1", "bc1", label="TypeX")
        cmap = {"TypeX": "#ff0000"}
        fig = plot_single_slab_scatter(df, cmap=cmap)
        plt.close(fig)

    def test_exclude_labels(self):
        rng = np.random.RandomState(7)
        df = _make_cells(50, SLAB, PLANE, "B1", "bc1", rng=rng)
        df["supercluster_term_label"] = rng.choice(["Keep", "Drop"], 50)
        fig = plot_single_slab_scatter(df, exclude_labels={"Drop"})
        ax = fig.axes[0]
        plotted_labels = {c.get_label() for c in ax.collections}
        assert "Drop" not in plotted_labels
        plt.close(fig)

    def test_empty_df(self):
        """Should produce a figure even when all coords are NaN."""
        df = _make_cells(10, SLAB, PLANE, "B1", "bc1")
        df["x_slab_mm"] = np.nan
        fig = plot_single_slab_scatter(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_slab_scatter (integration)
# ---------------------------------------------------------------------------

class TestPlotSlabScatter:
    def test_all_blocks_present(self):
        """All 9 blocks under QM24.50.002.CX.49 / Plane01 should appear."""
        df = _make_multi_block_df()
        figs = plot_slab_scatter(df)
        assert len(figs) == 1
        key = (SLAB, PLANE)
        assert key in figs
        plt.close("all")

    def test_metadata_merge(self):
        """slab/plane columns come from metadata_df via barcode join."""
        rng = np.random.RandomState(0)
        # df has no slab/plane columns
        df = pd.DataFrame({
            "x_slab_mm": rng.uniform(0, 5, 30),
            "y_slab_mm": rng.uniform(0, 5, 30),
            "barcode": ["bc1"] * 15 + ["bc2"] * 15,
            "supercluster_term_label": "T",
        })
        metadata = pd.DataFrame({
            "barcode": ["bc1", "bc2"],
            "slab": [SLAB, SLAB],
            "slab_plane": [PLANE, PLANE],
        })
        figs = plot_slab_scatter(df, metadata_df=metadata)
        assert (SLAB, PLANE) in figs
        plt.close("all")

    def test_metadata_merge_missing_barcode(self):
        """Barcodes absent from metadata get NaN slab/plane => dropped."""
        rng = np.random.RandomState(0)
        df = pd.DataFrame({
            "x_slab_mm": rng.uniform(0, 5, 30),
            "y_slab_mm": rng.uniform(0, 5, 30),
            "barcode": ["bc1"] * 10 + ["bc2"] * 10 + ["bc_unknown"] * 10,
            "supercluster_term_label": "T",
        })
        metadata = pd.DataFrame({
            "barcode": ["bc1", "bc2"],  # bc_unknown missing
            "slab": [SLAB, SLAB],
            "slab_plane": [PLANE, PLANE],
        })
        figs = plot_slab_scatter(df, metadata_df=metadata)
        # bc_unknown rows have NaN slab/plane -> dropped by groupby
        # Only 20 of 30 rows survive
        assert (SLAB, PLANE) in figs
        plt.close("all")
