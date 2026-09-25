from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
# Configuration
# ---------------------------------------------------------

SIGMA_LEVEL = 15
MIN_AREA = 3


# ---------------------------------------------------------
# Find FITS files
# ---------------------------------------------------------

fits_files = sorted(RAW_DIR.glob("*.fits"))

if not fits_files:
    raise FileNotFoundError(
        f"No FITS files found in {RAW_DIR}"
    )

fits_path = fits_files[0]

print("=" * 70)
print("DIGANTARA CANDIDATE BOUNDING BOX GENERATION")
print("=" * 70)

print(f"\nFile       : {fits_path.name}")
print(f"Sigma      : {SIGMA_LEVEL}σ")
print(f"Minimum area: {MIN_AREA} pixels")


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

print("\nEstimating background...")

background = gaussian_filter(
    image,
    sigma=25
)


# ---------------------------------------------------------
# Background-subtracted residual
# ---------------------------------------------------------

residual = image - background

valid = residual[
    np.isfinite(residual)
]


# ---------------------------------------------------------
# Robust noise estimation
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

print(
    f"Residual median : "
    f"{median_residual:.6f}"
)

print(
    f"Robust sigma    : "
    f"{robust_sigma:.6f}"
)

print(
    f"Threshold       : "
    f"{threshold:.6f}"
)


# ---------------------------------------------------------
# Binary detection mask
# ---------------------------------------------------------

candidate_mask = (
    residual > threshold
)


# ---------------------------------------------------------
# Connected components
# ---------------------------------------------------------

structure = np.ones(
    (3, 3),
    dtype=np.uint8
)

labeled, number = label(
    candidate_mask,
    structure=structure
)

objects = find_objects(
    labeled
)


# ---------------------------------------------------------
# Extract candidate objects
# ---------------------------------------------------------

candidates = []


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

    # Coordinates of component pixels
    yy, xx = np.where(region)

    centroid_x = (
        x_min + np.mean(xx)
    )

    centroid_y = (
        y_min + np.mean(yy)
    )

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    candidates.append({
        "id": object_id,
        "area": area,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": width,
        "height": height,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y
    })


# ---------------------------------------------------------
# Print statistics
# ---------------------------------------------------------

print("\nCandidate statistics")
print("-" * 70)

print(
    f"Initial components : "
    f"{number:,}"
)

print(
    f"Candidates after "
    f"area filtering    : "
    f"{len(candidates):,}"
)


if candidates:

    areas = np.array(
        [c["area"] for c in candidates]
    )

    widths = np.array(
        [c["width"] for c in candidates]
    )

    heights = np.array(
        [c["height"] for c in candidates]
    )

    print(
        f"Minimum area       : "
        f"{areas.min()}"
    )

    print(
        f"Maximum area       : "
        f"{areas.max()}"
    )

    print(
        f"Median area        : "
        f"{np.median(areas):.1f}"
    )

    print(
        f"Median width       : "
        f"{np.median(widths):.1f}"
    )

    print(
        f"Median height      : "
        f"{np.median(heights):.1f}"
    )


# ---------------------------------------------------------
# Display image
# ---------------------------------------------------------

finite_image = image[
    np.isfinite(image)
]

display_min = np.percentile(
    finite_image,
    1
)

display_max = np.percentile(
    finite_image,
    99.5
)


fig, ax = plt.subplots(
    figsize=(16, 10)
)


ax.imshow(
    image,
    cmap="gray",
    vmin=display_min,
    vmax=display_max,
    origin="upper"
)


# ---------------------------------------------------------
# Draw candidate boxes
# ---------------------------------------------------------

for candidate in candidates:

    x = candidate["x_min"]
    y = candidate["y_min"]

    width = candidate["width"]
    height = candidate["height"]

    rectangle = Rectangle(
        (
            x,
            y
        ),
        width,
        height,
        fill=False,
        edgecolor="red",
        linewidth=0.7
    )

    ax.add_patch(
        rectangle
    )


# ---------------------------------------------------------
# Plot settings
# ---------------------------------------------------------

ax.set_title(
    f"Candidate Bounding Boxes — "
    f"{SIGMA_LEVEL}σ, "
    f"Area ≥ {MIN_AREA}"
)

ax.set_xlabel(
    "X pixel"
)

ax.set_ylabel(
    "Y pixel"
)


plt.tight_layout()


# ---------------------------------------------------------
# Save visualization
# ---------------------------------------------------------

output_path = (
    VIS_DIR /
    "candidate_bounding_boxes.png"
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
print("CANDIDATE BOX GENERATION COMPLETE")
print("=" * 70)