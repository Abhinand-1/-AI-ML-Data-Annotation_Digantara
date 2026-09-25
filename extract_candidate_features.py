from pathlib import Path

import numpy as np
import pandas as pd

from astropy.io import fits
from scipy.ndimage import gaussian_filter, label, find_objects


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SIGMA_LEVEL = 15
MIN_AREA = 3
BOX_MARGIN = 5


# ---------------------------------------------------------
# Select first FITS
# ---------------------------------------------------------

fits_files = sorted(RAW_DIR.glob("*.fits"))

if not fits_files:
    raise FileNotFoundError(
        f"No FITS files found in {RAW_DIR}"
    )

fits_path = fits_files[0]

print("=" * 70)
print("DIGANTARA CANDIDATE FEATURE EXTRACTION")
print("=" * 70)

print(f"\nFile        : {fits_path.name}")
print(f"Sigma       : {SIGMA_LEVEL}σ")
print(f"Minimum area: {MIN_AREA}")
print(f"Box margin  : {BOX_MARGIN}px")


# ---------------------------------------------------------
# Read FITS
# ---------------------------------------------------------

with fits.open(
    fits_path,
    memmap=False
) as hdul:

    image = np.asarray(
        hdul[0].data,
        dtype=np.float32
    )

    image = np.squeeze(image)


# ---------------------------------------------------------
# Estimate background
# ---------------------------------------------------------

background = gaussian_filter(
    image,
    sigma=25
)


# ---------------------------------------------------------
# Residual
# ---------------------------------------------------------

residual = image - background

valid = residual[
    np.isfinite(residual)
]


# ---------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------

median_residual = np.median(valid)

mad = np.median(
    np.abs(
        valid - median_residual
    )
)

robust_sigma = 1.4826 * mad


# ---------------------------------------------------------
# Detection threshold
# ---------------------------------------------------------

threshold = (
    median_residual
    + SIGMA_LEVEL * robust_sigma
)


print("\nDetection statistics")
print("-" * 70)

print(f"Median residual : {median_residual:.6f}")
print(f"Robust sigma    : {robust_sigma:.6f}")
print(f"Threshold       : {threshold:.6f}")


# ---------------------------------------------------------
# Detection mask
# ---------------------------------------------------------

mask = residual > threshold


# ---------------------------------------------------------
# Connected components
# ---------------------------------------------------------

structure = np.ones(
    (3, 3),
    dtype=np.uint8
)

labeled, number = label(
    mask,
    structure=structure
)

objects = find_objects(
    labeled
)


# ---------------------------------------------------------
# Extract features
# ---------------------------------------------------------

records = []


for object_id, object_slice in enumerate(
    objects,
    start=1
):

    if object_slice is None:
        continue

    region = (
        labeled[object_slice]
        == object_id
    )

    area = np.count_nonzero(
        region
    )

    if area < MIN_AREA:
        continue

    y_slice, x_slice = object_slice

    x_min = x_slice.start
    x_max = x_slice.stop - 1

    y_min = y_slice.start
    y_max = y_slice.stop - 1


    # -----------------------------------------------------
    # Component pixels
    # -----------------------------------------------------

    yy, xx = np.where(region)

    pixel_y = y_min + yy
    pixel_x = x_min + xx

    values = residual[
        pixel_y,
        pixel_x
    ]


    # -----------------------------------------------------
    # Basic intensity features
    # -----------------------------------------------------

    peak_residual = float(
        np.max(values)
    )

    mean_residual = float(
        np.mean(values)
    )

    integrated_residual = float(
        np.sum(values)
    )


    # -----------------------------------------------------
    # Centroid
    # -----------------------------------------------------

    weights = np.maximum(
        values,
        0
    )

    weight_sum = np.sum(weights)

    if weight_sum > 0:

        centroid_x = float(
            np.sum(
                pixel_x * weights
            )
            / weight_sum
        )

        centroid_y = float(
            np.sum(
                pixel_y * weights
            )
            / weight_sum
        )

    else:

        centroid_x = float(
            np.mean(pixel_x)
        )

        centroid_y = float(
            np.mean(pixel_y)
        )


    # -----------------------------------------------------
    # Bounding box
    # -----------------------------------------------------

    width = (
        x_max
        - x_min
        + 1
    )

    height = (
        y_max
        - y_min
        + 1
    )

    aspect_ratio = (
        width / height
    )


    # -----------------------------------------------------
    # Expanded box
    # -----------------------------------------------------

    expanded_x_min = max(
        0,
        x_min - BOX_MARGIN
    )

    expanded_y_min = max(
        0,
        y_min - BOX_MARGIN
    )

    expanded_x_max = min(
        image.shape[1] - 1,
        x_max + BOX_MARGIN
    )

    expanded_y_max = min(
        image.shape[0] - 1,
        y_max + BOX_MARGIN
    )

    expanded_width = (
        expanded_x_max
        - expanded_x_min
        + 1
    )

    expanded_height = (
        expanded_y_max
        - expanded_y_min
        + 1
    )


    # -----------------------------------------------------
    # Record
    # -----------------------------------------------------

    records.append({

        "file":
            fits_path.name,

        "component_id":
            object_id,

        "area":
            area,

        "width":
            width,

        "height":
            height,

        "aspect_ratio":
            aspect_ratio,

        "peak_residual":
            peak_residual,

        "mean_residual":
            mean_residual,

        "integrated_residual":
            integrated_residual,

        "centroid_x":
            centroid_x,

        "centroid_y":
            centroid_y,

        "x_min":
            x_min,

        "y_min":
            y_min,

        "x_max":
            x_max,

        "y_max":
            y_max,

        "expanded_x_min":
            expanded_x_min,

        "expanded_y_min":
            expanded_y_min,

        "expanded_x_max":
            expanded_x_max,

        "expanded_y_max":
            expanded_y_max,

        "expanded_width":
            expanded_width,

        "expanded_height":
            expanded_height,

        "sigma_level":
            SIGMA_LEVEL,

        "threshold":
            threshold
    })


# ---------------------------------------------------------
# DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(
    records
)


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("\nCandidate feature summary")
print("-" * 70)

print(
    f"Number of candidates: "
    f"{len(df):,}"
)

if not df.empty:

    print("\nArea:")
    print(
        df["area"].describe()
    )

    print("\nPeak residual:")
    print(
        df["peak_residual"].describe()
    )

    print("\nIntegrated residual:")
    print(
        df["integrated_residual"].describe()
    )


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

output_path = (
    OUTPUT_DIR /
    "candidate_features_first_image.csv"
)

df.to_csv(
    output_path,
    index=False
)


print("\nSaved:")
print(output_path)

print("\n" + "=" * 70)
print("CANDIDATE FEATURE EXTRACTION COMPLETE")
print("=" * 70)