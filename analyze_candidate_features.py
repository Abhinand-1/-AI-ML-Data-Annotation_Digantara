from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "output"
    / "candidate_features_first_image.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
    / "visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Load candidate data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


print("=" * 70)
print("DIGANTARA CANDIDATE FEATURE ANALYSIS")
print("=" * 70)

print(f"\nInput file:")
print(INPUT_FILE)

print(f"\nNumber of candidates: {len(df):,}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")


# ---------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("FEATURE SUMMARY")
print("-" * 70)

features = [
    "area",
    "width",
    "height",
    "aspect_ratio",
    "peak_residual",
    "mean_residual",
    "integrated_residual"
]

print(
    df[features].describe().round(3)
)


# =========================================================
# 1. AREA DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["area"],
    bins=50
)

plt.xlabel("Component area (pixels)")
plt.ylabel("Number of candidates")
plt.title("Candidate Area Distribution")

plt.tight_layout()

path = OUTPUT_DIR / "candidate_area_distribution.png"

plt.savefig(path, dpi=200)

plt.close()

print(f"\nSaved: {path}")


# =========================================================
# 2. PEAK RESIDUAL DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["peak_residual"],
    bins=50
)

plt.xlabel("Peak residual")
plt.ylabel("Number of candidates")
plt.title("Peak Residual Distribution")

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "candidate_peak_distribution.png"
)

plt.savefig(path, dpi=200)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 3. INTEGRATED RESIDUAL DISTRIBUTION
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["integrated_residual"],
    bins=50
)

plt.xlabel("Integrated residual")
plt.ylabel("Number of candidates")
plt.title("Integrated Residual Distribution")

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "candidate_integrated_distribution.png"
)

plt.savefig(path, dpi=200)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 4. AREA VS PEAK RESIDUAL
# =========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["area"],
    df["peak_residual"],
    s=12,
    alpha=0.6
)

plt.xlabel("Component area (pixels)")
plt.ylabel("Peak residual")
plt.title("Area vs Peak Residual")

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "area_vs_peak_residual.png"
)

plt.savefig(path, dpi=200)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 5. AREA VS INTEGRATED RESIDUAL
# =========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["area"],
    df["integrated_residual"],
    s=12,
    alpha=0.6
)

plt.xlabel("Component area (pixels)")
plt.ylabel("Integrated residual")
plt.title("Area vs Integrated Residual")

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "area_vs_integrated_residual.png"
)

plt.savefig(path, dpi=200)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 6. WIDTH VS HEIGHT
# =========================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    df["width"],
    df["height"],
    s=12,
    alpha=0.6
)

plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.title("Candidate Width vs Height")

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "width_vs_height.png"
)

plt.savefig(path, dpi=200)

plt.close()

print(f"Saved: {path}")


# =========================================================
# 7. ASPECT RATIO
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["aspect_ratio"],
    bins=50
)

plt.xlabel("Aspect ratio (width / height)")
plt.ylabel("Number of candidates")
plt.title("Candidate Aspect Ratio Distribution")

plt.tight_layout()

path = (
    OUTPUT_DIR
    / "aspect_ratio_distribution.png"
)

plt.savefig(path, dpi=200)

plt.close()

print(f"Saved: {path}")


# =========================================================
# Strongest candidates
# =========================================================

print("\n" + "-" * 70)
print("TOP 20 CANDIDATES BY PEAK RESIDUAL")
print("-" * 70)

top_peak = df.sort_values(
    "peak_residual",
    ascending=False
).head(20)

print(
    top_peak[
        [
            "component_id",
            "area",
            "width",
            "height",
            "aspect_ratio",
            "peak_residual",
            "integrated_residual"
        ]
    ].to_string(index=False)
)


# =========================================================
# Strongest candidates by integrated signal
# =========================================================

print("\n" + "-" * 70)
print("TOP 20 CANDIDATES BY INTEGRATED RESIDUAL")
print("-" * 70)

top_integrated = df.sort_values(
    "integrated_residual",
    ascending=False
).head(20)

print(
    top_integrated[
        [
            "component_id",
            "area",
            "width",
            "height",
            "aspect_ratio",
            "peak_residual",
            "integrated_residual"
        ]
    ].to_string(index=False)
)


# =========================================================
# Complete
# =========================================================

print("\n" + "=" * 70)
print("CANDIDATE FEATURE ANALYSIS COMPLETE")
print("=" * 70)