from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.ndimage import gaussian_filter, label, find_objects


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "output"
VIS_DIR = OUTPUT_DIR / "visualizations"

VIS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# First FITS file
# ---------------------------------------------------------

fits_files = sorted(RAW_DIR.glob("*.fits"))

if not fits_files:
    raise FileNotFoundError(
        f"No FITS files found in {RAW_DIR}"
    )

fits_path = fits_files[0]

print("=" * 70)
print("DIGANTARA COMPONENT SIZE ANALYSIS")
print("=" * 70)

print(f"\nFile: {fits_path.name}")


# ---------------------------------------------------------
# Read image
# ---------------------------------------------------------

with fits.open(fits_path, memmap=False) as hdul:

    image = np.asarray(
        hdul[0].data,
        dtype=np.float32
    )

    image = np.squeeze(image)


# ---------------------------------------------------------
# Background subtraction
# ---------------------------------------------------------

background = gaussian_filter(
    image,
    sigma=25
)

residual = image - background

valid = residual[np.isfinite(residual)]


# ---------------------------------------------------------
# Robust noise estimate
# ---------------------------------------------------------

median_residual = np.median(valid)

mad = np.median(
    np.abs(valid - median_residual)
)

robust_sigma = 1.4826 * mad


# ---------------------------------------------------------
# Test three thresholds
# ---------------------------------------------------------

sigma_levels = [10, 15, 20]


for sigma_level in sigma_levels:

    threshold = (
        median_residual +
        sigma_level * robust_sigma
    )

    mask = residual > threshold

    structure = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    labeled, number = label(
        mask,
        structure=structure
    )

    objects = find_objects(labeled)

    areas = []

    for object_id, object_slice in enumerate(
        objects,
        start=1
    ):

        if object_slice is None:
            continue

        region = (
            labeled[object_slice] == object_id
        )

        area = np.count_nonzero(region)

        areas.append(area)

    areas = np.asarray(
        areas,
        dtype=np.int32
    )

    print("\n" + "-" * 70)
    print(f"{sigma_level}σ threshold")
    print("-" * 70)

    print(f"Threshold          : {threshold:.4f}")
    print(f"Total components   : {len(areas):,}")

    if len(areas) == 0:
        continue

    print(f"Minimum area       : {areas.min()}")
    print(f"Maximum area       : {areas.max()}")
    print(f"Mean area          : {areas.mean():.2f}")
    print(f"Median area        : {np.median(areas):.2f}")

    print("\nComponent counts:")

    for minimum_area in [
        1,
        2,
        3,
        5,
        10,
        20,
        50
    ]:

        count = np.count_nonzero(
            areas >= minimum_area
        )

        print(
            f"Area >= {minimum_area:2d}: "
            f"{count:,}"
        )


    # -----------------------------------------------------
    # Histogram
    # -----------------------------------------------------

    plt.figure(figsize=(10, 6))

    clipped_areas = np.clip(
        areas,
        1,
        100
    )

    plt.hist(
        clipped_areas,
        bins=np.arange(
            1,
            102,
            1
        )
    )

    plt.xlabel("Component area (pixels)")
    plt.ylabel("Number of components")

    plt.title(
        f"Component Size Distribution — {sigma_level}σ"
    )

    plt.tight_layout()

    output_path = (
        VIS_DIR /
        f"component_area_{sigma_level}sigma.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved histogram:")
    print(output_path)


print("\n" + "=" * 70)
print("COMPONENT ANALYSIS COMPLETE")
print("=" * 70)