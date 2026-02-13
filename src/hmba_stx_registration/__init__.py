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

from .st_to_block import register_cells_to_blockface
from .st_to_slab import barcode_transform_to_slab, process_barcode

__all__ = [
    "register_cells_to_blockface",
    "barcode_transform_to_slab",
    "process_barcode",
]
