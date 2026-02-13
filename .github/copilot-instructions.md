# HMBA Spatial TX Registration Agent Instructions

This document provides guidance for AI agents working on the `hmba-stx-registration` codebase.

## Overview

This is a Python pipeline for spatial transcriptomics (STX) data registration. It takes microscope images and associated data, handles various alignment edge cases, and produces affine transforms to map microscope coordinates to "slab" and "block" coordinates.

The core logic is organized into modules under `src/hmba_stx_registration/`:
- `st_to_block/registration.py`: Handles registration from ST data to a block.
- `block_sections/registration.py`: Manages registration for block sections.
- `slab_coords.py`: Likely contains logic related to the slab coordinate system.

## Key Concepts & Data Flow

The main goal is to generate a JSON file containing a 2D affine transformation matrix. This matrix transforms ST microscope coordinates (in µm, origin at bottom-left) to slab coordinates (in mm, origin at top-left).

- **Inputs**: The pipeline relies on specific input files with naming conventions like `{section_name}_mapping_for_registration_{yyyymmdd}.csv`. See the `README.md` for a full list.
- **Outputs**: The primary output is a JSON file, `{section_name}_coarse_transform_to_slab_mm_{yyyymmdd}.json`, which contains the transformation matrix. Other outputs include QC images and run manifests.

## Development Workflow

1.  **Installation**: Set up a virtual environment and install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Testing**: This project currently lacks a test suite. When adding new features, please include corresponding tests in the `tests/` directory.

## Code Conventions

- **Coordinate Systems**: Pay close attention to the different coordinate systems, units, and origins mentioned in the `README.md`. The transformation logic is sensitive to these details.
- **File Naming**: The pipeline heavily relies on file naming conventions for inputs and outputs, incorporating section names and dates (e.g., `{section_name}_..._{yyyymmdd}.json`). Maintain these conventions.
- **Edge Cases**: The `README.md` documents several critical edge cases, such as missing blockface images or sections split into multiple chunks. Ensure that any modifications or new features correctly handle these scenarios. For example, when a section is split, multiple transforms are generated, one for each `subset_label`.
