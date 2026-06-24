"""
Block image processing for the HMBA spatial-TX registration pipeline.

This sub-package provides helpers for interactively segmenting the OCT
(green resin) region and the embedded tissue from blockface microscope
images, and saving RGBA cutouts that downstream registration consumes.

The main entry point is :func:`process_slab`, which walks a slab folder
and produces one processed PNG per block specimen.
"""

from .background_registration import (
    magic_wand_mask,
    register_translations_via_magic_wand,
)
from .block_segmentation import (
    annotate_two_points,
    crop_image,
    crop_to_oct,
    find_block_image_paths,
    get_bbox_from_mask,
    get_oct_mask,
    load_normalized_image,
    normalize_exposure_lab,
    segment_slab_registered,
    rgba_from_rgb_and_mask,
    sam_mask_from_bbox,
    save_rgba_image,
    segment_green_oct,
    segment_tissue,
    show_annotation_qc,
)

__all__ = [
    "annotate_two_points",
    "crop_image",
    "crop_to_oct",
    "find_block_image_paths",
    "get_bbox_from_mask",
    "get_oct_mask",
    "load_normalized_image",
    "magic_wand_mask",
    "normalize_exposure_lab",
    "segment_slab_registered",
    "register_translations_via_magic_wand",
    "rgba_from_rgb_and_mask",
    "sam_mask_from_bbox",
    "save_rgba_image",
    "segment_green_oct",
    "segment_tissue",
    "show_annotation_qc",
]
