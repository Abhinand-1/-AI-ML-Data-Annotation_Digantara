import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# DIGANTARA CANDIDATE RANKING V2
#
# Baseline:
#   Detection = 15 sigma
#   Minimum component area = 3 pixels
#
# V2:
#   Signal + morphology + concentration
# ============================================================

BASE_DIR = r"C:\Users\Abhinand\DIGATRA_CODE_FILES"

FITS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "1a600998-b97c-4307-8632-6fcf573621ff.fits"
)

# If your FITS is stored somewhere else, change only FITS_FILE.

INPUT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "candidate_features_first_image.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

VIS_DIR = os.path.join(
    OUTPUT_DIR,
    "visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


print("=" * 75)
print("DIGANTARA CANDIDATE RANKING V2")
print("=" * 75)


# ============================================================
# 1. LOAD CANDIDATE FEATURES
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"\nCandidates loaded: {len(df)}")


required = [
    "component_id",
    "area",
    "width",
    "height",
    "aspect_ratio",
    "peak_residual",
    "mean_residual",
    "integrated_residual",
    "centroid_x",
    "centroid_y",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# 2. BASIC MORPHOLOGY
# ============================================================

# Bounding-box occupancy.
#
# A component occupying more of its bounding box is more compact
# than a sparse/irregular component.

df["bbox_area"] = (
    df["width"] *
    df["height"]
)

df["compactness"] = (
    df["area"] /
    df["bbox_area"]
)

# Width/height consistency.

df["width_height_ratio"] = (
    df["width"] /
    df["height"]
)

# Distance of aspect ratio from a square.

df["aspect_deviation"] = (
    np.abs(df["aspect_ratio"] - 1.0)
)


# ============================================================
# 3. SIGNAL CONCENTRATION
# ============================================================

# Peak fraction:
#
# High values mean a larger fraction of total signal is
# concentrated in the brightest pixel.
#
# Low values mean signal is distributed over more pixels.

df["peak_fraction"] = (
    df["peak_residual"] /
    df["integrated_residual"].replace(0, np.nan)
)

df["peak_fraction"] = (
    df["peak_fraction"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# Mean signal relative to peak.

df["mean_to_peak"] = (
    df["mean_residual"] /
    df["peak_residual"].replace(0, np.nan)
)

df["mean_to_peak"] = (
    df["mean_to_peak"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# 4. EQUIVALENT RADIUS
# ============================================================

df["equivalent_radius"] = np.sqrt(
    df["area"] / np.pi
)


# ============================================================
# 5. PERCENTILE / RANK SCORES
# ============================================================
#
# Unlike V1, these scores are based on percentile rank.
# Therefore we don't get many candidates artificially clipped
# to exactly 1.0.

def percentile_score(series):
    return series.rank(
        method="average",
        pct=True
    )


df["peak_score"] = percentile_score(
    df["peak_residual"]
)

df["integrated_score"] = percentile_score(
    df["integrated_residual"]
)

df["area_score"] = percentile_score(
    df["area"]
)


# ============================================================
# 6. MORPHOLOGY SCORES
# ============================================================

# Aspect ratio score.
#
# Maximum score occurs around aspect ratio = 1.

df["shape_score"] = np.exp(
    -(
        (df["aspect_ratio"] - 1.0)
        / 0.35
    ) ** 2
)


# Compactness is directly normalized.

compact_min = df["compactness"].min()
compact_max = df["compactness"].max()

if compact_max > compact_min:

    df["compactness_score"] = (
        (df["compactness"] - compact_min)
        /
        (compact_max - compact_min)
    )

else:

    df["compactness_score"] = 1.0


# Peak concentration score.

df["concentration_score"] = percentile_score(
    df["peak_fraction"]
)


# Mean-to-peak score.

df["mean_peak_score"] = percentile_score(
    df["mean_to_peak"]
)


# ============================================================
# 7. FINAL SCORE
# ============================================================
#
# Signal remains dominant.
#
# 30% peak signal
# 30% integrated signal
# 10% area
# 10% shape
# 10% compactness
# 10% concentration
#

df["candidate_score_v2"] = (
    0.30 * df["peak_score"]
    +
    0.30 * df["integrated_score"]
    +
    0.10 * df["area_score"]
    +
    0.10 * df["shape_score"]
    +
    0.10 * df["compactness_score"]
    +
    0.10 * df["concentration_score"]
)


# ============================================================
# 8. RANK
# ============================================================

df = df.sort_values(
    "candidate_score_v2",
    ascending=False
).reset_index(drop=True)

df["rank_v2"] = (
    np.arange(len(df)) + 1
)


# ============================================================
# 9. PRIORITY GROUPS
# ============================================================

def priority(score):

    if score >= 0.90:
        return "PRIORITY_1"

    elif score >= 0.75:
        return "PRIORITY_2"

    elif score >= 0.50:
        return "PRIORITY_3"

    else:
        return "LOW_PRIORITY"


df["priority"] = (
    df["candidate_score_v2"]
    .apply(priority)
)


# ============================================================
# 10. SAVE COMPLETE DATASET
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "candidate_ranked_v2.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 11. SAVE TOP 50
# ============================================================

top_columns = [
    "rank_v2",
    "component_id",
    "priority",
    "candidate_score_v2",
    "area",
    "width",
    "height",
    "aspect_ratio",
    "compactness",
    "peak_residual",
    "integrated_residual",
    "peak_fraction",
    "mean_to_peak",
    "equivalent_radius",
    "centroid_x",
    "centroid_y",
]

top50 = df[top_columns].head(50)

top50_file = os.path.join(
    OUTPUT_DIR,
    "top_50_candidates_v2.csv"
)

top50.to_csv(
    top50_file,
    index=False
)


# ============================================================
# 12. PRINT SUMMARY
# ============================================================

print("\n" + "-" * 75)
print("PRIORITY SUMMARY")
print("-" * 75)

print(
    df["priority"]
    .value_counts()
    .sort_index()
)


print("\n" + "-" * 75)
print("TOP 20 CANDIDATES — V2")
print("-" * 75)

print(
    top50.head(20).to_string(
        index=False
    )
)


# ============================================================
# 13. SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["candidate_score_v2"],
    bins=40
)

plt.xlabel(
    "Candidate score V2"
)

plt.ylabel(
    "Number of candidates"
)

plt.title(
    "Candidate Score Distribution — V2"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VIS_DIR,
        "candidate_score_v2_distribution.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 14. SCORE VS PEAK
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["candidate_score_v2"],
    df["peak_residual"],
    alpha=0.6
)

plt.xlabel(
    "Candidate score V2"
)

plt.ylabel(
    "Peak residual"
)

plt.title(
    "Candidate Score V2 vs Peak Residual"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VIS_DIR,
        "score_v2_vs_peak.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 15. SCORE VS INTEGRATED RESIDUAL
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["candidate_score_v2"],
    df["integrated_residual"],
    alpha=0.6
)

plt.xlabel(
    "Candidate score V2"
)

plt.ylabel(
    "Integrated residual"
)

plt.title(
    "Candidate Score V2 vs Integrated Residual"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VIS_DIR,
        "score_v2_vs_integrated.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 16. CONCENTRATION ANALYSIS
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["area"],
    df["peak_fraction"],
    alpha=0.6
)

plt.xlabel(
    "Component area (pixels)"
)

plt.ylabel(
    "Peak / Integrated residual"
)

plt.title(
    "Signal Concentration vs Component Area"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VIS_DIR,
        "area_vs_peak_fraction.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 17. COMPACTNESS ANALYSIS
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["compactness"],
    df["candidate_score_v2"],
    alpha=0.6
)

plt.xlabel(
    "Component compactness"
)

plt.ylabel(
    "Candidate score V2"
)

plt.title(
    "Compactness vs Candidate Score"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VIS_DIR,
        "compactness_vs_score_v2.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 18. TOP 100 SPATIAL DISTRIBUTION
# ============================================================

top100 = df.head(100)

plt.figure(figsize=(14, 9))

plt.scatter(
    top100["centroid_x"],
    top100["centroid_y"],
    s=25,
    alpha=0.8
)

plt.gca().invert_yaxis()

plt.xlabel("X pixel")
plt.ylabel("Y pixel")

plt.title(
    "Top 100 Candidate Locations — V2"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VIS_DIR,
        "top_100_candidates_v2_locations.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 19. FINAL
# ============================================================

print("\n" + "=" * 75)
print("RANKING V2 COMPLETE")
print("=" * 75)

print("\nSaved:")
print(
    f"  {output_file}"
)

print(
    f"  {top50_file}"
)

print("\nVisualizations saved in:")
print(
    f"  {VIS_DIR}"
)