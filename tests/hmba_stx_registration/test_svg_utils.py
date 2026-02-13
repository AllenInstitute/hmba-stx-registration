"""Tests for hmba_stx_registration.svg_utils."""

import math

import numpy as np
import pytest

from hmba_stx_registration.svg_utils import (
    aggregate_affine,
    get_scale_affine,
    get_translation_affine,
    parse_svg_transform,
)


# ---- get_scale_affine ----

class TestGetScaleAffine:
    def test_identity_when_ones(self):
        result = get_scale_affine(1.0, 1.0)
        np.testing.assert_array_equal(result, np.eye(3))

    def test_scales_diagonal(self):
        result = get_scale_affine(2.0, 3.0)
        expected = np.diag([2.0, 3.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_modifies_existing_affine(self):
        aff = np.eye(3)
        aff[0, 2] = 10  # translation
        result = get_scale_affine(5.0, 7.0, affine=aff)
        assert result[0, 0] == 5.0
        assert result[1, 1] == 7.0
        assert result[0, 2] == 10  # unchanged
        assert result is aff  # in-place


# ---- get_translation_affine ----

class TestGetTranslationAffine:
    def test_basic(self):
        result = get_translation_affine(10.0, 20.0)
        expected = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_in_place(self):
        aff = np.eye(3)
        result = get_translation_affine(5.0, 6.0, affine=aff)
        assert result is aff
        assert aff[0, 2] == 5.0
        assert aff[1, 2] == 6.0


# ---- parse_svg_transform ----

class TestParseSvgTransform:
    def test_translate(self):
        result = parse_svg_transform("translate(10, 20)")
        expected = get_translation_affine(10.0, 20.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale(self):
        result = parse_svg_transform("scale(2, 3)")
        expected = get_scale_affine(2.0, 3.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate_origin(self):
        result = parse_svg_transform("rotate(90)")
        c, s = math.cos(math.radians(90)), math.sin(math.radians(90))
        expected = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_matrix(self):
        result = parse_svg_transform("matrix(1,0,0,1,100,200)")
        # SVG matrix(a,b,c,d,e,f) → [[a,c,e],[b,d,f],[0,0,1]]
        expected = np.array([[1, 0, 100], [0, 1, 200], [0, 0, 1]], dtype=float)
        np.testing.assert_array_almost_equal(result, expected)

    def test_chained_transforms(self):
        result = parse_svg_transform("translate(10,0) scale(2,2)")
        # translate first, then scale → composed as I @ T @ S
        T = get_translation_affine(10.0, 0.0)
        S = get_scale_affine(2.0, 2.0)
        expected = T @ S
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_string(self):
        result = parse_svg_transform("")
        np.testing.assert_array_equal(result, np.eye(3))


# ---- aggregate_affine ----

class TestAggregateAffine:
    def test_identity_case(self):
        result = aggregate_affine(1, 0, 0, 1, 1, np.eye(3))
        np.testing.assert_array_almost_equal(result, np.eye(3))

    def test_translation_and_scale(self):
        result = aggregate_affine(1, 10, 20, 2, 3, np.eye(3))
        pt = np.array([1, 1, 1])
        out = result @ pt
        # scale first: (2, 3), then translate: (10+2, 20+3) = (12, 23)
        np.testing.assert_array_almost_equal(out[:2], [12, 23])

    def test_with_downsample(self):
        result = aggregate_affine(2, 0, 0, 1, 1, np.eye(3))
        pt = np.array([1, 1, 1])
        out = result @ pt
        # downsample(2) applied first → (2,2), then scale(1,1) → (2,2), translate(0,0) → (2,2)
        np.testing.assert_array_almost_equal(out[:2], [2, 2])
