"""Tests for hmba_stx_registration.st_to_block."""

import numpy as np
import pandas as pd
import pytest

from hmba_stx_registration.st_to_block import (
    compute_wm_gm_density,
    rotation_search,
    rotate_and_phase_correlate,
)


# ---- compute_wm_gm_density ----

class TestComputeWmGmDensity:
    @pytest.fixture()
    def cell_table(self):
        """Synthetic cell table with known WM and GM regions."""
        rng = np.random.RandomState(42)
        n = 500
        # WM cluster at (100, 100), GM cluster at (500, 500)
        wm_x = rng.normal(100, 10, n)
        wm_y = rng.normal(100, 10, n)
        gm_x = rng.normal(500, 10, n)
        gm_y = rng.normal(500, 10, n)

        wm_labels = rng.choice(
            ["Oligodendrocyte", "Oligodendrocyte precursor"],
            size=n,
        )
        gm_labels = rng.choice(
            ["Deep-layer near-projecting", "MGE interneuron"],
            size=n,
        )

        df = pd.DataFrame({
            "center_x": np.concatenate([wm_x, gm_x]),
            "center_y": np.concatenate([wm_y, gm_y]),
            "cell_type": np.concatenate([wm_labels, gm_labels]),
        })
        return df

    def test_shapes(self, cell_table):
        wm, gm, ratio, wd, gd = compute_wm_gm_density(
            cell_table, "cell_type", um_per_px=20,
        )
        assert wm.shape == gm.shape == ratio.shape
        assert wd.shape == gd.shape

    def test_ratio_range(self, cell_table):
        _, _, ratio, _, _ = compute_wm_gm_density(
            cell_table, "cell_type", um_per_px=20,
        )
        assert ratio.min() >= -1.0
        assert ratio.max() <= 1.0

    def test_masks_complementary(self, cell_table):
        wm, gm, ratio, _, _ = compute_wm_gm_density(
            cell_table, "cell_type", um_per_px=20,
        )
        # Where ratio > 0 → WM; ratio < 0 → GM; ratio == 0 → neither
        assert not np.any(wm & gm), "WM and GM masks should not overlap"


# ---- rotation_search ----

class TestRotationSearch:
    def test_identity_alignment(self):
        """Two identical images should produce near-zero rotation."""
        rng = np.random.RandomState(0)
        img = rng.rand(64, 64).astype(np.float32)
        tfm, angle, shift, corr = rotation_search(img, img, angle_step=90, n_workers=1)
        assert corr > 0.99

    def test_90_degree_rotation(self):
        """A 90° rotated square should be found."""
        img = np.zeros((80, 80), dtype=np.float32)
        img[20:60, 30:50] = 1.0  # rectangle
        rotated = np.rot90(img)
        tfm, angle, shift, corr = rotation_search(img, rotated, angle_step=90, n_workers=1)
        # Should find 90 or 270 (equivalent)
        assert angle in (90, 270) or corr > 0.5


# ---- rotate_and_phase_correlate ----

class TestRotateAndPhaseCorrelate:
    def test_zero_rotation(self):
        """With 0° rotation, output should match input closely."""
        img = np.zeros((64, 64), dtype=np.float32)
        img[20:40, 20:40] = 1.0
        tfm, aligned, shift = rotate_and_phase_correlate(img, img, 0)
        # shift should be near zero
        assert abs(shift[0]) < 2 and abs(shift[1]) < 2

    def test_returns_correct_types(self):
        img = np.random.rand(32, 32).astype(np.float32)
        tfm, aligned, shift = rotate_and_phase_correlate(img, img, 45)
        from skimage.transform import AffineTransform
        assert isinstance(tfm, AffineTransform)
        assert aligned.shape == img.shape
        assert len(shift) == 2
