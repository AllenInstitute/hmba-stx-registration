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
import pandas as pd
import json
from pathlib import Path
from typing import Union

__all__ = [
    "register_cells_to_blockface",
    "barcode_transform_to_slab",
    "process_barcode",
    "get_metadata_df",
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
    def __init__(self, barcode: str, base_barcodes_path: Union[str, Path]):
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

    @cached_property
    def cells_table(self, date: str = None) -> pd.DataFrame:
        cells_path = Path('')
        if date:
            cells_path = self.barcode_path / f"{self.specimen_name}_mapping_for_registration_{date}.csv"
        if not cells_path.exists() or date is None:
            cells_path = next(self.barcode_path.glob(f"{self.specimen_name}_mapping_for_registration*.csv"))
        if not cells_path.exists():
            raise FileNotFoundError(f"Cells file not found: {cells_path}")
        return pd.read_csv(cells_path, index_col=0)
    
