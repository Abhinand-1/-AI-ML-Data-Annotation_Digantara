from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter


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
print("DIGANTARA LOCAL BACKGROUND ANALYSIS")
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
# Estimate large-scale background
# ---------------------------------------------------------

print("\nEstimating local background...")

background = gaussian_filter(
    image,
    sigma=25
)


# ---------------------------------------------------------
# Calculate residual
# ---------------------------------------------------------

residual = image - background


# ---------------------------------------------------------
# Residual statistics
# ---------------------------------------------------------

valid = residual[np.isfinite(residual)]

median_residual = np.median(valid)

mad = np.median(
    np.abs(valid - median_residual)
)

robust_sigma = 1.4826 * mad


print("\nResidual statistics")
print("-" * 70)

print(f"Residual median : {median_residual:.6f}")
print(f"MAD             : {mad:.6f}")
print(f"Robust sigma    : {robust_sigma:.6f}")


# ---------------------------------------------------------
# Candidate threshold
# ---------------------------------------------------------

threshold = (
    median_residual +
    5 * robust_sigma
)

print("\nCandidate threshold")
print("-" * 70)

print(f"5-sigma residual threshold: {threshold:.6f}")


# ---------------------------------------------------------
# Candidate pixels
# ---------------------------------------------------------

candidate = residual > threshold

candidate_count = np.count_nonzero(candidate)

print("\nCandidate pixels")
print("-" * 70)

print(f"Candidate pixels: {candidate_count:,}")
print(
    f"Percentage      : "
    f"{candidate_count / image.size * 100:.6f}%"
)


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


# Original
low = np.percentile(
    image[np.isfinite(image)],
    1
)

high = np.percentile(
    image[np.isfinite(image)],
    99.5
)

axes[0].imshow(
    image,
    cmap="gray",
    vmin=low,
    vmax=high,
    origin="upper"
)

axes[0].set_title("Original")
axes[0].axis("off")


# Background
axes[1].imshow(
    background,
    cmap="gray",
    origin="upper"
)

axes[1].set_title("Estimated Background")
axes[1].axis("off")


# Residual
res_low = np.percentile(
    valid,
    1
)

res_high = np.percentile(
    valid,
    99.5
)

axes[2].imshow(
    residual,
    cmap="gray",
    vmin=res_low,
    vmax=res_high,
    origin="upper"
)

axes[2].set_title("Background-subtracted Residual")
axes[2].axis("off")


plt.tight_layout()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_path = (
    VIS_DIR /
    "local_background_analysis.png"
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
print("LOCAL BACKGROUND ANALYSIS COMPLETE")
print("=" * 70)