from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


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
print("DIGANTARA BACKGROUND / NOISE ANALYSIS")
print("=" * 70)

print(f"\nFile: {fits_path.name}")


# ---------------------------------------------------------
# Read image
# ---------------------------------------------------------

with fits.open(fits_path, memmap=False) as hdul:

    image = hdul[0].data

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.squeeze(image)


# ---------------------------------------------------------
# Valid pixels
# ---------------------------------------------------------

valid = image[np.isfinite(image)]

if valid.size == 0:
    raise ValueError("No valid pixels found.")


# ---------------------------------------------------------
# Robust background estimation
# ---------------------------------------------------------

background = np.median(valid)

mad = np.median(
    np.abs(valid - background)
)

robust_sigma = 1.4826 * mad


# ---------------------------------------------------------
# Detection threshold
# ---------------------------------------------------------

threshold_5sigma = (
    background +
    5 * robust_sigma
)

threshold_10sigma = (
    background +
    10 * robust_sigma
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\nBackground statistics")
print("-" * 70)

print(f"Median background : {background:.4f}")
print(f"MAD               : {mad:.4f}")
print(f"Robust sigma      : {robust_sigma:.4f}")

print("\nDetection thresholds")
print("-" * 70)

print(f"5-sigma threshold : {threshold_5sigma:.4f}")
print(f"10-sigma threshold: {threshold_10sigma:.4f}")


# ---------------------------------------------------------
# Candidate pixels
# ---------------------------------------------------------

candidate_5 = image > threshold_5sigma
candidate_10 = image > threshold_10sigma

count_5 = np.count_nonzero(candidate_5)
count_10 = np.count_nonzero(candidate_10)

total = image.size

print("\nCandidate pixels")
print("-" * 70)

print(
    f"> 5 sigma : "
    f"{count_5:,} "
    f"({count_5 / total * 100:.6f}%)"
)

print(
    f"> 10 sigma: "
    f"{count_10:,} "
    f"({count_10 / total * 100:.6f}%)"
)


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

low = np.percentile(valid, 1)
high = np.percentile(valid, 99.9)

plt.figure(figsize=(12, 8))

plt.imshow(
    image,
    cmap="gray",
    vmin=low,
    vmax=high,
    origin="upper"
)

plt.colorbar(label="Pixel value")

plt.title(
    "Background / Noise Analysis"
)

plt.xlabel("X pixel")
plt.ylabel("Y pixel")

plt.tight_layout()

output_path = (
    VIS_DIR /
    "background_analysis_first_image.png"
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
print("BACKGROUND ANALYSIS COMPLETE")
print("=" * 70)