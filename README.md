# HMBA Spatial TX Registration
This document describes the inputs, outputs, and assumptions of the pipeline. The outputs are intended to be stable for downstream processing.

## Inputs  
### Directory structure
* `{barcode_number}`
* `{donor_name}_section_metadata.csv`

### Files dependences
* `{section_name}_mapping_for_registration_{yyyymmdd}.csv`
* `{section_name}_column_names_for_registration_{yyyymmdd}.json`


## Edge cases
1. Corresponding blockface image section doesn't exist for barcode section 
   * Get closest section from the same set
        * Example:
            * barcode section = QM24.50.002.CX.51.02.01.03
            * blockface img section = QM24.50.002.CX.51.02.01.02
1. Corresponding blockface image set doesn't exist for barcode set
   * Get closest set from the same block. If there are two closest sets (e.g. if set 03 is missing, 01 and 05 are equally close, pick the visually most similar one)
        * Mappings for Barbossa (missing <- replacement)
            ```
            QM24.50.002.CX.45.05.01 <- QM24.50.002.CX.45.05.03.02
            QM24.50.002.CX.45.04.05 <- QM24.50.002.CX.45.04.03.02
            QM24.50.002.CX.46.04.03 <- QM24.50.002.CX.46.04.01.02
            QM24.50.002.CX.47.07.04 <- QM24.50.002.CX.47.07.03.02
            QM24.50.002.CX.48.04.03 <- QM24.50.002.CX.48.04.05.02
            QM24.50.002.CX.48.05.03 <- QM24.50.002.CX.48.05.02.02
            QM24.50.002.CX.50.05.01 <- QM24.50.002.CX.50.05.03.02
            QM24.50.002.CX.50.06.05 <- QM24.50.002.CX.50.06.03.02
    * This is currently implemented by duplicating the image files with the renamed section name and does not have any code that treats these differently.
1. Multiple sections per set (if ST reimaging is performed)
1. Section split into multiple chunks with each chunk requiring multiple independent transforms to the slab
    * If blockface is split:
        * Subset blockface denoted by {section_name}_{subset_label}.png
        * Subset cells by with clustering and label clusters to map to the subset_label of the blockfaces
        * Each blockface subset has its own transform which gets applied to each subset of cells
    * Intersect mask with cells to map cells to `subset_label`
    * Transform for each `subset_label` stored in the transform_to_slab_mm.json file

## Outputs
1. Run manifest 
`{section_name}_run_manifest_{yyyymmdd}.json`
    * `schema_version` — manifest schema version (currently `"1.0"`); bump when the schema changes
    * `hmba_stx_registration_version` — package version string (e.g. `"0.1.0"`)
    * `date` — run date as `yyyymmdd`
    * `specimen_name`
    * `input_files` — list of input filenames
    * `output_files` — list of output filenames
    * `args` — dictionary of parameters used in the run

    * Example file:
    ```json
    {
        "schema_version": "1.0",
        "hmba_stx_registration_version": "0.1",
        "date": "20260217",
        "specimen_name": "QM24.50.002.CX.45.05.01.02",
        "input_files": [
            "QM24.50.002.CX.45.05.01.02_mapping_for_registration_20250929.csv",
            "QM24.50.002.CX.45.05.01.02_column_names_for_registration_20250929.json"
        ],
        "output_files": [
            "QM24.50.002.CX.45.05.01.02_registration_block_qc_20260217.png",
            "QM24.50.002.CX.45.05.01.02_coarse_registration_slab_qc_20260217.png",
            "QM24.50.002.CX.45.05.01.02_coarse_registration_slab_qc_20260217.png",
            "QM24.50.002.CX.45.05.01.02_coarse_transform_to_slab_mm_20260217.json"
        ],
        "args": {
            "um_per_px": 20,
            "table_label": "supercluster_term_label"
        }
    }
    ```
1. Transforms to slabs (mm unit) `{section_name}_coarse_transform_to_slab_mm_{yyyymmdd}.json`
    * subset_label: accounts for sections split into multiple chunks each with independent an transform. Cells are mapped to `subset_label` column in `{section_name}_coarse_transform_slab_coordinates_{yyyymmdd}.csv`. The mask associated with each subset label is included as `{section_name}_subset_mask_{subset_label}.npy`
       * Valid values: int count from 0 or 'nan' if section does not have subset labels
    * 2D affine
        * **input**: microscope coordinates (µm units, origin bottom-left, x,y order)
        * **output**: slab coordinates (mm units, origin top-left, x,y order)
        * usage:
        ```
        def transform_coordinates(coordinates: np.array, transform_matrix: np.array):
            coords = np.hstack((coordinates, np.ones((coordinates.shape[0],1))))
            transformed_coords = transform_matrix @ coords.T
            return transformed_coords.T
    * Example file
    ```
    [
        {
            "source": "QM24.50.002.CX.51.01.05.02",
            "subset_label": 0,
            "source_unit": "micrometer",
            "source_origin": "bottomleft",
            "target_unit": "millimeter",
            "target_origin": "topleft",
            "axis_order": "xy",
            "transform": [
                [
                    0.03803254687122138,
                    0.0356922983550712,
                    1164.409722814227
                ],
                [
                    0.03691804592876794,
                    -0.03581437850414539,
                    2117.136143270175
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        {
            "source": "QM24.50.002.CX.51.01.05.02",
            "subset_label": 1,
            "source_unit": "micrometer",
            "source_origin": "bottomleft",
            "target_unit": "millimeter",
            "target_origin": "topleft",
            "axis_order": "xy",
            "transform": [
                [
                    0.02510127262955611,
                    0.04484479844578686,
                    1177.050781225488
                ],
                [
                    0.04668343192673359,
                    -0.023357095431405458,
                    1960.2173668250891
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        }
    ]
1. QC registration to slab `{section_name}_coarse_registration_slab_qc_{yyyymmdd}.png` or `{section_name}_{subset_label)_coarse_registration_slab_qc_{yyyymmdd}.png`
1. QC registration to block `{section_name}_registration_block_qc_{yyyymmdd}.png`
1. OPTIONAL: subset_label mapping to cells csv `{section_name}_subset_label.csv`