"""Tests for hmba_stx_registration.st_to_slab."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hmba_stx_registration import Specimen
from hmba_stx_registration.st_to_slab import (
    barcode_transform_to_slab,
    get_specimen_name_from_barcode,
    plot_coarse_registration_slab_qc,
    process_barcode,
)
from hmba_stx_registration.utils import transform_coordinates


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
        assert name is None  # ambiguous -> warning + None


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
        assert len(result[0]["transform"]) == 3  # 3x3

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

    def test_closest_picks_nearest_section(self, tmp_path):
        """With sections 01 and 05 available, section 03 should match closer one."""
        metadata = pd.DataFrame({
            "barcode": ["BC004"],
            "specimen_set_name": ["QM.50.002.CX.45.01.01"],
            "section": ["03"],
        })
        specimen_name = "QM.50.002.CX.45.01.01.03"
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        np.save(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy", np.eye(3))

        bf_affines = {
            "QM.50.002.CX.45.01.01.01": np.eye(3) * 1,
            "QM.50.002.CX.45.01.01.05": np.eye(3) * 2,
        }
        bf_affines["QM.50.002.CX.45.01.01.01"][2, 2] = 1.0
        bf_affines["QM.50.002.CX.45.01.01.05"][2, 2] = 1.0

        result = barcode_transform_to_slab(
            "BC004", metadata, bf_affines, transforms_path, um_per_px=20,
        )
        assert result is not None
        assert len(result) == 1


# ---- Transform JSON output spec (README Output #2) ----

class TestTransformJsonSchema:
    """Validate the transform manifest structure against the README spec."""

    @pytest.fixture()
    def single_transform(self, tmp_path):
        metadata = pd.DataFrame({
            "barcode": ["BC010"],
            "specimen_set_name": ["QM.50.002.CX.51.01.05"],
            "section": ["02"],
        })
        specimen_name = "QM.50.002.CX.51.01.05.02"
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        np.save(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy", np.eye(3))
        bf_affines = {specimen_name: np.eye(3)}
        return barcode_transform_to_slab("BC010", metadata, bf_affines, transforms_path, um_per_px=20)

    def test_is_list(self, single_transform):
        assert isinstance(single_transform, list)

    def test_entry_has_required_keys(self, single_transform):
        required_keys = {
            "source", "subset_label", "source_unit", "source_origin",
            "target_unit", "target_origin", "axis_order", "transform",
        }
        for entry in single_transform:
            assert required_keys.issubset(entry.keys())
            assert entry["source_unit"] == "micrometer"
            assert entry["source_origin"] == "bottomleft"
            assert entry["target_unit"] == "millimeter"
            assert entry["target_origin"] == "topleft"
            assert entry["axis_order"] == "xy"

    def test_transform_is_3x3(self, single_transform):
        for entry in single_transform:
            tfm = np.array(entry["transform"])
            assert tfm.shape == (3, 3)

    def test_last_row_is_001(self, single_transform):
        for entry in single_transform:
            tfm = np.array(entry["transform"])
            np.testing.assert_array_equal(tfm[2], [0, 0, 1])

    def test_subset_label_valid(self, single_transform):
        """subset_label must be int >= 0 or the string 'nan'."""
        for entry in single_transform:
            sl = entry["subset_label"]
            assert sl == "nan" or (isinstance(sl, int) and sl >= 0)


# ---- Transform chain math ----

class TestTransformChainMath:
    """Verify scale(1/um_per_px) -> st_to_bf -> bf_to_slab -> scale(um_per_px/1000)."""

    def test_identity_chain(self, tmp_path):
        """With identity affines the chain should just be scale(1/1000)."""
        um_per_px = 20
        metadata = pd.DataFrame({
            "barcode": ["BC020"],
            "specimen_set_name": ["QM.50.002.CX.51.01.05"],
            "section": ["02"],
        })
        specimen_name = "QM.50.002.CX.51.01.05.02"
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        np.save(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy", np.eye(3))
        bf_affines = {specimen_name: np.eye(3)}

        result = barcode_transform_to_slab(
            "BC020", metadata, bf_affines, transforms_path, um_per_px=um_per_px,
        )
        tfm = np.array(result[0]["transform"])

        # Expected: scale(um_per_px/1000) @ I @ I @ scale(1/um_per_px)
        #         = scale(1/1000)
        expected = np.diag([1 / 1000, 1 / 1000, 1.0])
        np.testing.assert_array_almost_equal(tfm, expected)

    def test_known_point_transform(self, tmp_path):
        """A point at (1000um, 2000um) through identity affines should map to (1mm, 2mm)."""
        um_per_px = 20
        metadata = pd.DataFrame({
            "barcode": ["BC021"],
            "specimen_set_name": ["QM.50.002.CX.51.01.05"],
            "section": ["02"],
        })
        specimen_name = "QM.50.002.CX.51.01.05.02"
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        np.save(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy", np.eye(3))
        bf_affines = {specimen_name: np.eye(3)}

        result = barcode_transform_to_slab(
            "BC021", metadata, bf_affines, transforms_path, um_per_px=um_per_px,
        )
        tfm = np.array(result[0]["transform"])

        coords_um = np.array([[1000.0, 2000.0]])
        coords_mm = transform_coordinates(coords_um, tfm)
        np.testing.assert_array_almost_equal(coords_mm[0, :2], [1.0, 2.0])


# ---- QC image outputs (README Outputs #3, #4) ----

class TestPlotCoarseRegistrationSlabQc:
    def test_creates_png(self, tmp_path):
        slab_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        coords_mm = np.random.rand(200, 2) * 2  # 200 points in [0,2] mm
        label_color = pd.Series(["#ff0000"] * 200)
        out = tmp_path / "qc.png"
        plot_coarse_registration_slab_qc(slab_img, coords_mm, label_color, 0.02, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_filename_convention(self, tmp_path):
        """Output name should follow {section}_{subset}_coarse_registration_slab_qc_{date}.png."""
        slab_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        coords_mm = np.random.rand(100, 2)
        label_color = pd.Series(["#00ff00"] * 100)
        fname = "QM24.50.002.CX.51.01.05.02_coarse_registration_slab_qc_20260213.png"
        out = tmp_path / fname
        plot_coarse_registration_slab_qc(slab_img, coords_mm, label_color, 0.02, out)
        assert out.exists()


# ---- Edge case 4: multi-subset transform manifest ----

class TestMultiSubsetTransformManifest:
    """Section split into multiple chunks -> independent transforms per subset."""

    def test_independent_transforms(self, tmp_path):
        """Each subset_label should have a different transform when bf_affines differ."""
        metadata = pd.DataFrame({
            "barcode": ["BC030"],
            "specimen_set_name": ["QM.50.002.CX.51.01.05"],
            "section": ["02"],
        })
        specimen_name = "QM.50.002.CX.51.01.05.02"
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        np.save(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy", np.eye(3))

        # Two different blockface affines for the two subsets
        bf0 = np.eye(3)
        bf1 = np.array([[1, 0, 100], [0, 1, 200], [0, 0, 1]], dtype=float)
        bf_affines = {
            f"{specimen_name}_0": bf0,
            f"{specimen_name}_1": bf1,
        }

        result = barcode_transform_to_slab(
            "BC030", metadata, bf_affines, transforms_path, um_per_px=20,
        )
        assert len(result) == 2
        assert result[0]["subset_label"] == 0
        assert result[1]["subset_label"] == 1
        # Transforms should differ because bf_affines differ
        assert result[0]["transform"] != result[1]["transform"]

    def test_subset_labels_sequential_from_zero(self, tmp_path):
        """subset_label values should be sequential integers starting from 0."""
        metadata = pd.DataFrame({
            "barcode": ["BC031"],
            "specimen_set_name": ["QM.50.002.CX.51.01.05"],
            "section": ["03"],
        })
        specimen_name = "QM.50.002.CX.51.01.05.03"
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        np.save(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy", np.eye(3))

        bf_affines = {
            f"{specimen_name}_0": np.eye(3),
            f"{specimen_name}_1": np.eye(3),
            f"{specimen_name}_2": np.eye(3),
        }

        result = barcode_transform_to_slab(
            "BC031", metadata, bf_affines, transforms_path, um_per_px=20,
        )
        labels = [r["subset_label"] for r in result]
        assert labels == [0, 1, 2]


# ---- process_barcode integration test (README Outputs #1-#4) ----

class TestProcessBarcode:
    """Integration test that process_barcode writes the expected output files."""

    @pytest.fixture()
    def barcode_env(self, tmp_path):
        """Set up a complete mock environment for process_barcode."""
        specimen_name = "QM.50.002.CX.51.01.05.02"
        barcode = "BARCODE001"
        date = "20260213"

        # Transforms dir with ST->BF affine + QC
        transforms_path = tmp_path / "transforms"
        transforms_path.mkdir()
        (transforms_path / "qc").mkdir()
        np.save(transforms_path / f"{specimen_name}_sptx_to_bf_affine.npy", np.eye(3))

        # Fake block QC image
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1])
        fig.savefig(transforms_path / "qc" / f"{specimen_name}_registration_qc.png")
        plt.close(fig)

        # Barcodes dir with mapping CSV
        barcodes_path = tmp_path / "barcodes"
        barcode_dir = barcodes_path / barcode
        barcode_dir.mkdir(parents=True)

        # Create a mapping CSV with required columns
        rng = np.random.RandomState(0)
        n = 50
        mapping_df = pd.DataFrame({
            "center_x": rng.rand(n) * 1000,
            "center_y": rng.rand(n) * 1000,
            "supercluster_term_name": rng.choice(["TypeA", "TypeB"], n),
        })
        mapping_df.to_csv(barcode_dir / f"{specimen_name}_mapping_for_registration_{date}.csv")

        # Create specimen metadata JSON for Specimen construction
        metadata_json = {
            "barcode": barcode,
            "donor": "QM.50.002",
            "division": "CX",
            "slab": "51",
            "block": "01",
            "set": "05",
            "section": "02",
        }
        with open(barcode_dir / f"{barcode}_specimen_metadata.json", "w") as f:
            json.dump(metadata_json, f)

        specimen = Specimen(barcode, barcodes_path)

        # Blockface affines and slab image
        bf_affines = {specimen_name: np.eye(3)}
        slab_imgs = {51: np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}

        return {
            "barcode": barcode,
            "specimen": specimen,
            "bf_affines": bf_affines,
            "slab_imgs": slab_imgs,
            "transforms_path": transforms_path,
            "barcodes_path": barcodes_path,
            "date": date,
            "specimen_name": specimen_name,
        }

    def test_output_files_created(self, barcode_env):
        env = barcode_env
        result = process_barcode(
            specimen=env["specimen"],
            bf_affines=env["bf_affines"],
            slab_imgs=env["slab_imgs"],
            transforms_path=env["transforms_path"],
            table_label="supercluster_term_name",
            um_per_px=20,
            date=env["date"],
            version="0.1.0",
            sync_to_s3=False,
        )
        assert result is not None
        results_dir = env["barcodes_path"] / env["barcode"] / f"registration_results_{env['date']}"
        assert results_dir.exists()

        sn = env["specimen_name"]
        dt = env["date"]

        # Output #1: transform manifest JSON
        tfm_path = results_dir / f"{sn}_coarse_transform_to_slab_mm_{dt}.json"
        assert tfm_path.exists()
        tfm_data = json.loads(tfm_path.read_text())
        assert isinstance(tfm_data, list)
        assert "source" in tfm_data[0]
        assert "subset_label" in tfm_data[0]
        assert "source_unit" in tfm_data[0]
        assert "source_origin" in tfm_data[0]
        assert "target_unit" in tfm_data[0]
        assert "target_origin" in tfm_data[0]
        assert "axis_order" in tfm_data[0]
        assert "transform" in tfm_data[0]

        # Output #2: run manifest JSON (new schema)
        run_path = results_dir / f"{sn}_run_manifest_{dt}.json"
        assert run_path.exists()
        run_data = json.loads(run_path.read_text())
        assert run_data["schema_version"] == "1.0"
        assert run_data["hmba_stx_registration_version"] == "0.1.0"
        assert run_data["date"] == dt
        assert run_data["specimen_name"] == sn
        assert isinstance(run_data["input_files"], list)
        assert isinstance(run_data["output_files"], list)
        assert isinstance(run_data["args"], dict)

        # Output #3: block QC (copied)
        block_qc = results_dir / f"{sn}_registration_block_qc_{dt}.png"
        assert block_qc.exists()

        # Output #4: slab QC
        slab_qc = results_dir / f"{sn}_coarse_registration_slab_qc_{dt}.png"
        assert slab_qc.exists()

        # slab QC filename should appear in result["output_files"]
        qc_filename = f"{sn}_coarse_registration_slab_qc_{dt}.png"
        assert qc_filename in result["output_files"]

    def test_run_manifest_lists_all_outputs(self, barcode_env):
        env = barcode_env
        process_barcode(
            specimen=env["specimen"],
            bf_affines=env["bf_affines"],
            slab_imgs=env["slab_imgs"],
            transforms_path=env["transforms_path"],
            table_label="supercluster_term_name",
            um_per_px=20,
            date=env["date"],
            version="0.1.0",
            sync_to_s3=False,
        )
        results_dir = env["barcodes_path"] / env["barcode"] / f"registration_results_{env['date']}"
        run_data = json.loads(
            (results_dir / f"{env['specimen_name']}_run_manifest_{env['date']}.json").read_text(),
        )
        # slab QC filename should be listed in the manifest
        sn = env["specimen_name"]
        dt = env["date"]
        qc_filename = f"{sn}_coarse_registration_slab_qc_{dt}.png"
        assert qc_filename in run_data["output_files"]

        # All output files mentioned in manifest should actually exist
        for fname in run_data["output_files"]:
            assert (results_dir / fname).exists(), f"Output file {fname} listed in manifest but missing"
