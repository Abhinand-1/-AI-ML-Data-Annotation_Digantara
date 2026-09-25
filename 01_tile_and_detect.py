from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"C:\Users\Abhinand\DIGATRA_CODE_FILES")

RAW_DIR = ROOT / "data" / "raw"

OUTPUT_DIR = ROOT / "output" / "final_annotation"

TILES_DIR = OUTPUT_DIR / "tiles"
IMAGE_TILES_DIR = TILES_DIR / "images"
PREMASK_DIR = TILES_DIR / "premasks"

METADATA_FILE = OUTPUT_DIR / "tile_metadata.csv"


# ============================================================
# PARAMETERS
# ============================================================

TILE_SIZE = 1024

BACKGROUND_SIGMA = 25

THRESHOLD_SIGMA = 15

MIN_COMPONENT_AREA = 3


# ============================================================
# CREATE DIRECTORIES
# ============================================================

IMAGE_TILES_DIR.mkdir(parents=True, exist_ok=True)
PREMASK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ FITS
# ============================================================

def read_fits(path):

    with fits.open(path, memmap=False) as hdul:

        image = hdul[0].data.astype(np.float32)

    if image.ndim != 2:

        raise ValueError(
            f"{path.name}: expected 2D image, "
            f"but got shape {image.shape}"
        )

    return image


# ============================================================
# ROBUST NOISE ESTIMATION
# ============================================================

def robust_sigma(image):

    median = np.median(image)

    mad = np.median(
        np.abs(image - median)
    )

    sigma = 1.4826 * mad

    return sigma


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess(image):

    # Estimate smooth local background
    background = gaussian_filter(
        image,
        sigma=BACKGROUND_SIGMA
    )

    # Remove background
    residual = image - background

    # Robust noise estimate
    median = np.median(residual)

    sigma = robust_sigma(residual)

    # Safety fallback
    if sigma <= 0:

        sigma = float(np.std(residual))

    # Detection threshold
    threshold = (
        median +
        THRESHOLD_SIGMA * sigma
    )

    # Binary detection
    binary = (
        residual > threshold
    ).astype(np.uint8)

    # 8-connected components
    number, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    for component_id in range(1, number):

        x, y, width, height, area = (
            stats[component_id]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        components.append(
            (
                component_id,
                x,
                y,
                width,
                height,
                area
            )
        )

    return (
        residual,
        labels,
        components,
        threshold,
        sigma
    )


# ============================================================
# PRE-CLASSIFICATION
# ============================================================

def pre_classify(width, height):

    """
    This is ONLY a preliminary class suggestion.

    1 = Blob
    2 = Streak

    Final masks must be visually checked.
    """

    major = max(width, height)

    minor = max(
        1,
        min(width, height)
    )

    aspect_ratio = (
        major / minor
    )

    if (
        aspect_ratio >= 3.0
        and major >= 8
    ):

        return 2

    return 1


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():

    all_metadata = []

    fits_files = sorted(
        RAW_DIR.glob("*.fits")
    )

    print("=" * 70)
    print("DIGANTARA IMAGE TILING AND CANDIDATE GENERATION")
    print("=" * 70)

    print(
        f"\nFound {len(fits_files)} FITS images."
    )

    if len(fits_files) != 10:

        print(
            "WARNING: Expected 10 raw FITS images."
        )

    for image_number, fits_path in enumerate(
        fits_files,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"[{image_number}/{len(fits_files)}] "
            f"{fits_path.name}"
        )

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = read_fits(
            fits_path
        )

        height, width = image.shape

        print(
            f"Original dimensions: "
            f"{width} x {height}"
        )

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        (
            residual,
            labels,
            components,
            threshold,
            sigma
        ) = preprocess(image)

        print(
            f"Robust sigma: {sigma:.6f}"
        )

        print(
            f"Detection threshold: "
            f"{threshold:.6f}"
        )

        print(
            f"Detected components: "
            f"{len(components)}"
        )

        # ----------------------------------------------------
        # CALCULATE TILE GRID
        # ----------------------------------------------------

        tile_rows = int(
            np.ceil(
                height / TILE_SIZE
            )
        )

        tile_cols = int(
            np.ceil(
                width / TILE_SIZE
            )
        )

        print(
            f"Tile grid: "
            f"{tile_rows} rows × "
            f"{tile_cols} columns"
        )

        print(
            f"Tiles for image: "
            f"{tile_rows * tile_cols}"
        )

        # ----------------------------------------------------
        # PAD IMAGE
        # ----------------------------------------------------

        padded_height = (
            tile_rows * TILE_SIZE
        )

        padded_width = (
            tile_cols * TILE_SIZE
        )

        pad_bottom = (
            padded_height - height
        )

        pad_right = (
            padded_width - width
        )

        padded_image = np.pad(
            image,
            (
                (0, pad_bottom),
                (0, pad_right)
            ),
            mode="edge"
        )

        padded_residual = np.pad(
            residual,
            (
                (0, pad_bottom),
                (0, pad_right)
            ),
            mode="constant"
        )

        # ----------------------------------------------------
        # GENERATE TILES
        # ----------------------------------------------------

        for row in range(tile_rows):

            for col in range(tile_cols):

                y0 = (
                    row *
                    TILE_SIZE
                )

                x0 = (
                    col *
                    TILE_SIZE
                )

                y1 = y0 + TILE_SIZE
                x1 = x0 + TILE_SIZE

                tile = padded_image[
                    y0:y1,
                    x0:x1
                ]

                residual_tile = (
                    padded_residual[
                        y0:y1,
                        x0:x1
                    ]
                )

                tile_id = (
                    f"{fits_path.stem}"
                    f"__r{row:02d}"
                    f"_c{col:02d}"
                )

                image_file = (
                    IMAGE_TILES_DIR /
                    f"{tile_id}.npy"
                )

                premask_file = (
                    PREMASK_DIR /
                    f"{tile_id}.npy"
                )

                # ------------------------------------------------
                # SAVE IMAGE TILE
                # ------------------------------------------------

                np.save(
                    image_file,
                    tile.astype(np.float32)
                )

                # ------------------------------------------------
                # CREATE EMPTY MASK
                #
                # 0 = background
                # 1 = blob
                # 2 = streak
                # ------------------------------------------------

                mask = np.zeros(
                    (
                        TILE_SIZE,
                        TILE_SIZE
                    ),
                    dtype=np.uint8
                )

                feature_count = 0

                # ------------------------------------------------
                # ADD DETECTED COMPONENTS
                # ------------------------------------------------

                for (
                    component_id,
                    cx,
                    cy,
                    comp_width,
                    comp_height,
                    area
                ) in components:

                    center_x = (
                        cx +
                        comp_width / 2
                    )

                    center_y = (
                        cy +
                        comp_height / 2
                    )

                    # Component belongs to this tile
                    if not (
                        x0 <= center_x < x1
                        and
                        y0 <= center_y < y1
                    ):
                        continue

                    ys, xs = np.where(
                        labels ==
                        component_id
                    )

                    local_x = xs - x0
                    local_y = ys - y0

                    valid = (
                        (local_x >= 0)
                        &
                        (local_x < TILE_SIZE)
                        &
                        (local_y >= 0)
                        &
                        (local_y < TILE_SIZE)
                    )

                    local_x = local_x[
                        valid
                    ]

                    local_y = local_y[
                        valid
                    ]

                    if len(local_x) == 0:
                        continue

                    class_id = pre_classify(
                        comp_width,
                        comp_height
                    )

                    mask[
                        local_y,
                        local_x
                    ] = class_id

                    feature_count += 1

                    aspect_ratio = (
                        max(
                            comp_width,
                            comp_height
                        )
                        /
                        max(
                            1,
                            min(
                                comp_width,
                                comp_height
                            )
                        )
                    )

                    all_metadata.append(
                        {
                            "source_file":
                                fits_path.name,

                            "tile_id":
                                tile_id,

                            "row":
                                row,

                            "col":
                                col,

                            "source_x0":
                                x0,

                            "source_y0":
                                y0,

                            "component_id":
                                int(
                                    component_id
                                ),

                            "area":
                                int(area),

                            "width":
                                int(
                                    comp_width
                                ),

                            "height":
                                int(
                                    comp_height
                                ),

                            "aspect_ratio":
                                float(
                                    aspect_ratio
                                ),

                            "preclass":
                                (
                                    "STREAK"
                                    if class_id == 2
                                    else "BLOB"
                                )
                        }
                    )

                # ------------------------------------------------
                # SAVE PRE-MASK
                # ------------------------------------------------

                np.save(
                    premask_file,
                    mask
                )

                # ------------------------------------------------
                # TILE METADATA
                # ------------------------------------------------

                all_metadata.append(
                    {
                        "source_file":
                            fits_path.name,

                        "tile_id":
                            tile_id,

                        "row":
                            row,

                        "col":
                            col,

                        "source_x0":
                            x0,

                        "source_y0":
                            y0,

                        "image_width":
                            width,

                        "image_height":
                            height,

                        "pad_right":
                            pad_right,

                        "pad_bottom":
                            pad_bottom,

                        "threshold":
                            float(
                                threshold
                            ),

                        "robust_sigma":
                            float(
                                sigma
                            ),

                        "feature_count":
                            feature_count,

                        "image_path":
                            str(
                                image_file
                            ),

                        "premask_path":
                            str(
                                premask_file
                            )
                    }
                )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = pd.DataFrame(
        all_metadata
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata.to_csv(
        METADATA_FILE,
        index=False
    )

    total_tiles = (
        len(fits_files)
        *
        70
    )

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Images processed: "
        f"{len(fits_files)}"
    )

    print(
        "Expected tiles approximately: "
        f"{total_tiles}"
    )

    print(
        f"Image tiles: "
        f"{IMAGE_TILES_DIR}"
    )

    print(
        f"Pre-masks: "
        f"{PREMASK_DIR}"
    )

    print(
        f"Metadata: "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()