from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.ndimage import gaussian_filter, label


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
print("DIGANTARA DETECTION THRESHOLD TEST")
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

print("\nEstimating background...")

background = gaussian_filter(
    image,
    sigma=25
)

residual = image - background

valid = residual[np.isfinite(residual)]


# ---------------------------------------------------------
# Robust noise
# ---------------------------------------------------------

median_residual = np.median(valid)

mad = np.median(
    np.abs(valid - median_residual)
)

robust_sigma = 1.4826 * mad

print(f"Residual median : {median_residual:.6f}")
print(f"Robust sigma    : {robust_sigma:.6f}")


# ---------------------------------------------------------
# Threshold levels
# ---------------------------------------------------------

sigma_levels = [
    5,
    8,
    10,
    15,
    20
]


# ---------------------------------------------------------
# Display figure
# ---------------------------------------------------------

fig, axes = plt.subplots(
    5,
    1,
    figsize=(14, 28)
)


for ax, sigma_level in zip(
    axes,
    sigma_levels
):

    threshold = (
        median_residual +
        sigma_level * robust_sigma
    )

    mask = residual > threshold

    # Connected components
    structure = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    labeled, number = label(
        mask,
        structure=structure
    )

    candidate_pixels = np.count_nonzero(mask)

    print("\n" + "-" * 60)
    print(f"{sigma_level} sigma")
    print(f"Threshold       : {threshold:.4f}")
    print(f"Candidate pixels: {candidate_pixels:,}")
    print(f"Components      : {number:,}")

    # Display residual mask
    ax.imshow(
        mask,
        cmap="gray",
        origin="upper"
    )

    ax.set_title(
        f"{sigma_level}σ | "
        f"Threshold={threshold:.2f} | "
        f"Pixels={candidate_pixels:,} | "
        f"Components={number:,}"
    )

    ax.axis("off")


plt.tight_layout()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_path = (
    VIS_DIR /
    "detection_threshold_comparison.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\n" + "=" * 70)
print("THRESHOLD TEST COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(output_path)