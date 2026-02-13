"""Tests for hmba_stx_registration.st_to_slab."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hmba_stx_registration.st_to_slab import (
    barcode_transform_to_slab,
    get_specimen_name_from_barcode,
)


# ---- get_specimen_name_from_barcode ----

class TestGetSpecimenNameFromBarcode:
    @pytest.fixture()
    def metadata(self):
        return pd.DataFrame({
            "barcode": ["BC001", "BC002"],
            "specimen_set_name": ["D.X.45.01.01", "D.X.45.01.02"],
            "section": ["03", "01"],
        })

    def test_found(self, metadata):
        name = get_specimen_name_from_barcode("BC001", metadata)
        assert name == "D.X.45.01.01.03"

    def test_not_found(self, metadata):
        name = get_specimen_name_from_barcode("MISSING", metadata)
        assert name is None

    def test_duplicate_barcode(self):
        df = pd.DataFrame({
            "barcode": ["BC001", "BC001"],
            "specimen_set_name": ["A", "B"],
            "section": ["01", "02"],
        })
        name = get_specimen_name_from_barcode("BC001", df)
        assert name is None  # ambiguous → warning + None


# ---- barcode_transform_to_slab ----

class TestBarcodeTransformToSlab:
    @pytest.fixture()
    def setup(self, tmp_path):
        """Create metadata, bf_affines, and a fake sptx_to_bf affine."""
        metadata = pd.DataFrame({
            "barcode": ["BC001"],
            "specimen_set_name": ["Q.D.45.01.01"],
            "section": ["02"],
        })
        specimen_name = "Q.D.45.01.01.02"

        # Save a dummy ST-to-BF affine
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        st_to_bf = np.eye(3) * 0.5
        st_to_bf[2, 2] = 1.0
        np.save(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy", st_to_bf)

        # Blockface affine (identity)
        bf_affines = {specimen_name: np.eye(3)}

        return metadata, bf_affines, transforms_path

    def test_single_piece(self, setup):
        metadata, bf_affines, transforms_path = setup
        result = barcode_transform_to_slab(
            "BC001", metadata, bf_affines, transforms_path, um_per_px=20,
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["source"] == "Q.D.45.01.01.02"
        assert result[0]["subset_label"] == "nan"
        assert len(result[0]["transform"]) == 3  # 3×3

    def test_missing_affine(self, tmp_path):
        metadata = pd.DataFrame({
            "barcode": ["BC999"],
            "specimen_set_name": ["Q.D.45.99.99"],
            "section": ["01"],
        })
        transforms_path = tmp_path / "empty"
        transforms_path.mkdir()
        result = barcode_transform_to_slab(
            "BC999", metadata, {}, transforms_path,
        )
        assert result is None

    def test_multi_subset(self, tmp_path):
        metadata = pd.DataFrame({
            "barcode": ["BC002"],
            "specimen_set_name": ["Q.D.46.02.03"],
            "section": ["01"],
        })
        specimen_name = "Q.D.46.02.03.01"
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        np.save(
            transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy",
            np.eye(3),
        )
        bf_affines = {
            f"{specimen_name}_0": np.eye(3),
            f"{specimen_name}_1": np.eye(3) * 2,
        }
        # Fix the second affine so [2,2]=1
        bf_affines[f"{specimen_name}_1"][2, 2] = 1.0

        result = barcode_transform_to_slab(
            "BC002", metadata, bf_affines, transforms_path, um_per_px=20,
        )
        assert result is not None
        assert len(result) == 2
        assert result[0]["subset_label"] == 0
        assert result[1]["subset_label"] == 1

    def test_closest_section_fallback(self, tmp_path):
        """When the exact section doesn't exist, use the closest one."""
        metadata = pd.DataFrame({
            "barcode": ["BC003"],
            "specimen_set_name": ["QM.50.002.CX.45.01.01"],
            "section": ["03"],
        })
        specimen_name = "QM.50.002.CX.45.01.01.03"
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        np.save(
            transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy",
            np.eye(3),
        )
        # bf_affines only has section 02, not 03
        bf_affines = {"QM.50.002.CX.45.01.01.02": np.eye(3)}

        result = barcode_transform_to_slab(
            "BC003", metadata, bf_affines, transforms_path, um_per_px=20,
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["source"] == specimen_name
