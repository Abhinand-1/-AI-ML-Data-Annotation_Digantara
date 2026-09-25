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
print("DIGANTARA CANDIDATE SHAPE INSPECTION")
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
# Robust statistics
# ---------------------------------------------------------

median_residual = np.median(valid)

mad = np.median(
    np.abs(valid - median_residual)
)

robust_sigma = 1.4826 * mad


# ---------------------------------------------------------
# Select threshold
# ---------------------------------------------------------

SIGMA_LEVEL = 15

threshold = (
    median_residual +
    SIGMA_LEVEL * robust_sigma
)

print(f"\nDetection level: {SIGMA_LEVEL}σ")
print(f"Threshold      : {threshold:.4f}")


# ---------------------------------------------------------
# Binary mask
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Build component list
# ---------------------------------------------------------

components = []

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

    y_slice, x_slice = object_slice

    y0 = y_slice.start
    y1 = y_slice.stop

    x0 = x_slice.start
    x1 = x_slice.stop

    yy, xx = np.where(region)

    centroid_y = y0 + np.mean(yy)
    centroid_x = x0 + np.mean(xx)

    components.append({
        "id": object_id,
        "area": area,
        "x": centroid_x,
        "y": centroid_y,
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1
    })


# ---------------------------------------------------------
# Sort by area
# ---------------------------------------------------------

components.sort(
    key=lambda c: c["area"],
    reverse=True
)


# ---------------------------------------------------------
# Select representative components
# ---------------------------------------------------------

# Large components
large = [
    c for c in components
    if c["area"] >= 10
]

# Medium components
medium = [
    c for c in components
    if 3 <= c["area"] < 10
]

# Small components
small = [
    c for c in components
    if c["area"] <= 2
]


print("\nComponent categories")
print("-" * 70)

print(f"Large   (>=10 px): {len(large):,}")
print(f"Medium  (3-9 px) : {len(medium):,}")
print(f"Small   (<=2 px) : {len(small):,}")


# ---------------------------------------------------------
# Select examples
# ---------------------------------------------------------

examples = []

examples.extend(
    large[:8]
)

examples.extend(
    medium[:8]
)

examples.extend(
    small[:8]
)

# Maximum 24 examples
examples = examples[:24]


# ---------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------

fig, axes = plt.subplots(
    4,
    6,
    figsize=(15, 10)
)

axes = axes.ravel()


half_size = 15


for ax, component in zip(
    axes,
    examples
):

    cx = int(round(component["x"]))
    cy = int(round(component["y"]))

    x0 = max(
        0,
        cx - half_size
    )

    x1 = min(
        image.shape[1],
        cx + half_size + 1
    )

    y0 = max(
        0,
        cy - half_size
    )

    y1 = min(
        image.shape[0],
        cy + half_size + 1
    )

    cutout = residual[
        y0:y1,
        x0:x1
    ]

    ax.imshow(
        cutout,
        cmap="gray",
        origin="upper"
    )

    ax.set_title(
        f"ID {component['id']} | "
        f"Area={component['area']}"
    )

    ax.axis("off")


# Hide unused axes

for ax in axes[len(examples):]:
    ax.axis("off")


fig.suptitle(
    f"Candidate Shape Inspection — {SIGMA_LEVEL}σ",
    fontsize=16
)

plt.tight_layout()


output_path = (
    VIS_DIR /
    "candidate_shape_inspection.png"
)

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.show()


print("\nSaved:")
print(output_path)

print("\n" + "=" * 70)
print("CANDIDATE SHAPE INSPECTION COMPLETE")
print("=" * 70)