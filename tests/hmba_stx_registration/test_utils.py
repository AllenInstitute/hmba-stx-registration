"""Tests for hmba_stx_registration.utils."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hmba_stx_registration.utils import (
    cluster_coordinates,
    create_run_manifest_json,
    generate_random_colormap,
    transform_coordinates,
)


# ---- generate_random_colormap ----

class TestGenerateRandomColormap:
    def test_returns_dict(self):
        labels = ["A", "B", "C"]
        cmap = generate_random_colormap(labels, seed=0)
        assert isinstance(cmap, dict)
        assert set(cmap.keys()) == set(labels)

    def test_hex_format(self):
        cmap = generate_random_colormap(["X"], seed=1)
        val = cmap["X"]
        assert val.startswith("#")
        assert len(val) == 7

    def test_reproducibility(self):
        a = generate_random_colormap(["A", "B"], seed=42)
        b = generate_random_colormap(["A", "B"], seed=42)
        assert a == b

    def test_different_seeds(self):
        a = generate_random_colormap(["A"], seed=0)
        b = generate_random_colormap(["A"], seed=99)
        assert a != b


# ---- transform_coordinates ----

class TestTransformCoordinates:
    def test_identity(self):
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = transform_coordinates(coords, np.eye(3))
        np.testing.assert_array_almost_equal(result[:, :2], coords)

    def test_translation(self):
        coords = np.array([[0.0, 0.0]])
        T = np.eye(3)
        T[0, 2] = 10
        T[1, 2] = 20
        result = transform_coordinates(coords, T)
        np.testing.assert_array_almost_equal(result[0, :2], [10, 20])

    def test_scale(self):
        coords = np.array([[1.0, 2.0]])
        S = np.diag([3.0, 4.0, 1.0])
        result = transform_coordinates(coords, S)
        np.testing.assert_array_almost_equal(result[0, :2], [3, 8])

    def test_combined(self):
        coords = np.array([[1.0, 1.0]])
        # scale(2,2) then translate(5,5)
        M = np.array([[2, 0, 5], [0, 2, 5], [0, 0, 1]], dtype=float)
        result = transform_coordinates(coords, M)
        np.testing.assert_array_almost_equal(result[0, :2], [7, 7])


# ---- cluster_coordinates ----

class TestClusterCoordinates:
    def test_two_clusters(self):
        rng = np.random.RandomState(0)
        n = 200
        # Cluster at (0, 0) and (0, 5000) — well separated
        top = pd.DataFrame({"x": rng.randn(n) * 10, "y": rng.randn(n) * 10})
        bottom = pd.DataFrame({"x": rng.randn(n) * 10, "y": rng.randn(n) * 10 + 5000})
        table = pd.concat([top, bottom], ignore_index=True)
        result = cluster_coordinates(table, coords_cols=["x", "y"])

        assert "subset_label" in result.columns
        assert set(result["subset_label"].unique()) == {0, 1}
        # Cluster 0 should be the top (lower y)
        assert result.loc[result["subset_label"] == 0, "y"].mean() < result.loc[result["subset_label"] == 1, "y"].mean()

    def test_returns_copy(self):
        rng = np.random.RandomState(1)
        n = 100
        table = pd.DataFrame({
            "x": np.concatenate([rng.randn(n) * 5, rng.randn(n) * 5]),
            "y": np.concatenate([rng.randn(n) * 5, rng.randn(n) * 5 + 5000]),
        })
        result = cluster_coordinates(table, coords_cols=["x", "y"])
        assert "subset_label" not in table.columns  # original untouched


# ---- create_run_manifest_json ----

class TestCreateRunManifest:
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "manifest.json"
            create_run_manifest_json(
                ver="0.1.0",
                date="20260213",
                specimen_name="QM24.50.002.CX.51.01.05.02",
                input_files=["input_a.csv", "input_b.json"],
                output_files=["/some/path/output.png"],
                args={"key": "value"},
                out_path=out,
            )
            data = json.loads(out.read_text())
            assert data["hmba_stx_registration_version"] == "0.1.0"
            assert data["date"] == "20260213"
            assert data["specimen_name"] == "QM24.50.002.CX.51.01.05.02"
            assert "input_a.csv" in data["input_files"]
            assert data["output_files"] == ["output.png"]
            assert data["args"] == {"key": "value"}

    def test_schema_version_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "m.json"
            create_run_manifest_json(
                ver="1.0.0", date="20260101",
                specimen_name="X", input_files=[], output_files=[],
                out_path=out,
            )
            data = json.loads(out.read_text())
            assert "schema_version" in data
            assert data["schema_version"] == "1.0"

    def test_version_stored_as_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "m.json"
            create_run_manifest_json(
                ver="0.2", date="20260101",
                specimen_name="X", input_files=[], output_files=[],
                out_path=out,
            )
            data = json.loads(out.read_text())
            assert isinstance(data["hmba_stx_registration_version"], str)

    def test_default_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "m.json"
            create_run_manifest_json(
                ver="1.0", date="20260101",
                specimen_name="X", input_files=[], output_files=[],
                out_path=out,
            )
            data = json.loads(out.read_text())
            assert data["args"] == {}

    def test_required_keys(self):
        """Manifest must contain all keys defined in the schema."""
        required = {
            "schema_version", "hmba_stx_registration_version",
            "date", "specimen_name",
            "input_files", "output_files", "args",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "m.json"
            create_run_manifest_json(
                ver="0.1.0", date="20260213",
                specimen_name="S", input_files=["a"], output_files=["b"],
                out_path=out,
            )
            data = json.loads(out.read_text())
            assert required == set(data.keys())

    def test_output_files_basenames_only(self):
        """Output files should be stored as basenames, not full paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "m.json"
            create_run_manifest_json(
                ver="0.1.0", date="20260213", specimen_name="S",
                input_files=["/long/path/to/input.csv"],
                output_files=["/long/path/to/output.png"],
                out_path=out,
            )
            data = json.loads(out.read_text())
            assert data["input_files"] == ["input.csv"]
            assert data["output_files"] == ["output.png"]
