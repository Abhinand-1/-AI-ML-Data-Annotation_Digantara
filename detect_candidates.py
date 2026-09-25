from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.ndimage import gaussian_filter, label, find_objects


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "output"
VIS_DIR = OUTPUT_DIR / "visualizations"

VIS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Select first FITS file
# ---------------------------------------------------------

fits_files = sorted(RAW_DIR.glob("*.fits"))

if not fits_files:
    raise FileNotFoundError(
        f"No FITS files found in {RAW_DIR}"
    )

fits_path = fits_files[0]

print("=" * 70)
print("DIGANTARA CANDIDATE DETECTION")
print("=" * 70)

print(f"\nFile: {fits_path.name}")


# ---------------------------------------------------------
# Read FITS
# ---------------------------------------------------------

with fits.open(fits_path, memmap=False) as hdul:

    image = hdul[0].data

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.squeeze(image)


# ---------------------------------------------------------
# Estimate background
# ---------------------------------------------------------

print("\nEstimating background...")

background = gaussian_filter(
    image,
    sigma=25
)


# ---------------------------------------------------------
# Calculate residual
# ---------------------------------------------------------

residual = image - background

valid = residual[np.isfinite(residual)]


# ---------------------------------------------------------
# Robust noise estimation
# ---------------------------------------------------------

median_residual = np.median(valid)

mad = np.median(
    np.abs(valid - median_residual)
)

robust_sigma = 1.4826 * mad


# ---------------------------------------------------------
# Detection threshold
# ---------------------------------------------------------

threshold = (
    median_residual +
    5 * robust_sigma
)

print("\nDetection parameters")
print("-" * 70)

print(f"Residual median : {median_residual:.6f}")
print(f"Robust sigma    : {robust_sigma:.6f}")
print(f"Threshold       : {threshold:.6f}")


# ---------------------------------------------------------
# Create binary candidate mask
# ---------------------------------------------------------

candidate_mask = residual > threshold

candidate_pixels = np.count_nonzero(
    candidate_mask
)

print("\nCandidate pixels")
print("-" * 70)

print(f"Candidate pixels: {candidate_pixels:,}")


# ---------------------------------------------------------
# Connected component labeling
# ---------------------------------------------------------

print("\nFinding connected components...")

structure = np.ones(
    (3, 3),
    dtype=np.uint8
)

labeled, num_objects = label(
    candidate_mask,
    structure=structure
)

print(f"Initial components: {num_objects:,}")


# ---------------------------------------------------------
# Find component slices
# ---------------------------------------------------------

objects = find_objects(labeled)


# ---------------------------------------------------------
# Analyze components
# ---------------------------------------------------------

candidates = []

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

    # Pixel coordinates
    y_slice, x_slice = object_slice

    y0 = y_slice.start
    y1 = y_slice.stop

    x0 = x_slice.start
    x1 = x_slice.stop

    # Centroid
    yy, xx = np.where(region)

    centroid_y = y0 + np.mean(yy)
    centroid_x = x0 + np.mean(xx)

    candidates.append({
        "id": object_id,
        "area": area,
        "x_min": x0,
        "x_max": x1 - 1,
        "y_min": y0,
        "y_max": y1 - 1,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y
    })


# ---------------------------------------------------------
# Component statistics
# ---------------------------------------------------------

areas = np.array(
    [c["area"] for c in candidates],
    dtype=np.int32
)

print("\nComponent statistics")
print("-" * 70)

print(f"Total components : {len(candidates):,}")

if len(areas) > 0:

    print(f"Minimum area     : {areas.min()}")
    print(f"Maximum area     : {areas.max()}")
    print(f"Median area      : {np.median(areas):.1f}")

    print(
        f"Components >= 2 pixels : "
        f"{np.count_nonzero(areas >= 2):,}"
    )

    print(
        f"Components >= 3 pixels : "
        f"{np.count_nonzero(areas >= 3):,}"
    )

    print(
        f"Components >= 5 pixels : "
        f"{np.count_nonzero(areas >= 5):,}"
    )

    print(
        f"Components >= 10 pixels: "
        f"{np.count_nonzero(areas >= 10):,}"
    )


# ---------------------------------------------------------
# Keep preliminary candidates
# ---------------------------------------------------------

MIN_AREA = 2

filtered_candidates = [
    c
    for c in candidates
    if c["area"] >= MIN_AREA
]

print("\nPreliminary filtering")
print("-" * 70)

print(f"Minimum area: {MIN_AREA}")
print(
    f"Remaining candidates: "
    f"{len(filtered_candidates):,}"
)


# ---------------------------------------------------------
# Create candidate visualization
# ---------------------------------------------------------

display_valid = image[
    np.isfinite(image)
]

low = np.percentile(
    display_valid,
    1
)

high = np.percentile(
    display_valid,
    99.5
)


plt.figure(figsize=(14, 9))

plt.imshow(
    image,
    cmap="gray",
    vmin=low,
    vmax=high,
    origin="upper"
)


# Plot candidate centers

if filtered_candidates:

    x = [
        c["centroid_x"]
        for c in filtered_candidates
    ]

    y = [
        c["centroid_y"]
        for c in filtered_candidates
    ]

    plt.scatter(
        x,
        y,
        s=8,
        facecolors="none",
        edgecolors="red",
        linewidths=0.5
    )


plt.title(
    "Candidate Sources — Preliminary Detection"
)

plt.xlabel("X pixel")
plt.ylabel("Y pixel")

plt.tight_layout()


output_path = (
    VIS_DIR /
    "candidate_detection_first_image.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


print("\nSaved:")
print(output_path)

print("\n" + "=" * 70)
print("CANDIDATE DETECTION COMPLETE")
print("=" * 70)