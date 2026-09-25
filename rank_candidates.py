import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DIGANTARA CANDIDATE RANKING
# Baseline: 15 sigma, minimum area = 3
# ============================================================

INPUT_FILE = (
    r"C:\Users\Abhinand\DIGATRA_CODE_FILES"
    r"\output\candidate_features_first_image.csv"
)

OUTPUT_DIR = (
    r"C:\Users\Abhinand\DIGATRA_CODE_FILES"
    r"\output"
)

VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("DIGANTARA CANDIDATE RANKING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load candidates
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nInput candidates: {len(df)}")

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

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# ------------------------------------------------------------
# 2. Robust normalization
# ------------------------------------------------------------
# Percentile normalization prevents a few extreme candidates
# from dominating the ranking.

def robust_normalize(series, low=5, high=95):
    lo = np.percentile(series, low)
    hi = np.percentile(series, high)

    if hi <= lo:
        return pd.Series(np.ones(len(series)), index=series.index)

    x = (series - lo) / (hi - lo)
    return x.clip(0, 1)


df["peak_score"] = robust_normalize(
    df["peak_residual"]
)

df["integrated_score"] = robust_normalize(
    df["integrated_residual"]
)

df["area_score"] = robust_normalize(
    df["area"]
)

# ------------------------------------------------------------
# 3. Shape / compactness score
# ------------------------------------------------------------
#
# Aspect ratio close to 1 is treated as more compact.
#
# IMPORTANT:
# This is only a morphology feature.
# It does NOT mean that aspect ratio ~1 proves a source
# is a real astronomical object.

df["shape_score"] = np.exp(
    -((df["aspect_ratio"] - 1.0) / 0.35) ** 2
)

# ------------------------------------------------------------
# 4. Size consistency
# ------------------------------------------------------------
#
# Extremely tiny components are more susceptible to
# pixel-level noise.
#
# We reward candidates above the median area without
# making large area alone decisive.

median_area = df["area"].median()

df["size_score"] = np.minimum(
    df["area"] / max(median_area, 1),
    3.0
) / 3.0

# ------------------------------------------------------------
# 5. Combined candidate score
# ------------------------------------------------------------
#
# Signal strength receives the largest contribution.
#
# Weights:
# peak residual       35%
# integrated residual 35%
# area                10%
# shape               10%
# size                10%
#

df["candidate_score"] = (
    0.35 * df["peak_score"]
    + 0.35 * df["integrated_score"]
    + 0.10 * df["area_score"]
    + 0.10 * df["shape_score"]
    + 0.10 * df["size_score"]
)

# ------------------------------------------------------------
# 6. Rank
# ------------------------------------------------------------

df = df.sort_values(
    "candidate_score",
    ascending=False
).reset_index(drop=True)

df["rank"] = np.arange(1, len(df) + 1)

# ------------------------------------------------------------
# 7. Candidate categories
# ------------------------------------------------------------

def classify(row):

    score = row["candidate_score"]

    if score >= 0.75:
        return "HIGH_SIGNAL"

    elif score >= 0.50:
        return "MEDIUM_SIGNAL"

    else:
        return "LOW_SIGNAL"


df["candidate_category"] = df.apply(
    classify,
    axis=1
)

# ------------------------------------------------------------
# 8. Save complete ranked dataset
# ------------------------------------------------------------

ranked_file = os.path.join(
    OUTPUT_DIR,
    "candidate_ranked.csv"
)

df.to_csv(
    ranked_file,
    index=False
)

# ------------------------------------------------------------
# 9. Save top 50
# ------------------------------------------------------------

top50_columns = [
    "rank",
    "component_id",
    "candidate_category",
    "candidate_score",
    "area",
    "width",
    "height",
    "aspect_ratio",
    "peak_residual",
    "integrated_residual",
    "centroid_x",
    "centroid_y",
]

top50 = df[top50_columns].head(50)

top50_file = os.path.join(
    OUTPUT_DIR,
    "top_50_candidates.csv"
)

top50.to_csv(
    top50_file,
    index=False
)

# ------------------------------------------------------------
# 10. Print summary
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("CANDIDATE CATEGORY SUMMARY")
print("-" * 70)

print(
    df["candidate_category"]
    .value_counts()
    .sort_index()
)

print("\n" + "-" * 70)
print("TOP 20 CANDIDATES")
print("-" * 70)

print(
    top50.head(20).to_string(index=False)
)

# ------------------------------------------------------------
# 11. Score distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    df["candidate_score"],
    bins=40
)

plt.xlabel("Candidate score")
plt.ylabel("Number of candidates")
plt.title("Candidate Score Distribution — 15σ")

plt.tight_layout()

score_plot = os.path.join(
    VIS_DIR,
    "candidate_score_distribution.png"
)

plt.savefig(
    score_plot,
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# 12. Score vs integrated residual
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["candidate_score"],
    df["integrated_residual"],
    alpha=0.6
)

plt.xlabel("Candidate score")
plt.ylabel("Integrated residual")
plt.title("Candidate Score vs Integrated Residual")

plt.tight_layout()

plot_file = os.path.join(
    VIS_DIR,
    "candidate_score_vs_integrated.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# 13. Score vs peak residual
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["candidate_score"],
    df["peak_residual"],
    alpha=0.6
)

plt.xlabel("Candidate score")
plt.ylabel("Peak residual")
plt.title("Candidate Score vs Peak Residual")

plt.tight_layout()

plot_file = os.path.join(
    VIS_DIR,
    "candidate_score_vs_peak.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# 14. Spatial distribution of top candidates
# ------------------------------------------------------------

top_n = df.head(100)

plt.figure(figsize=(14, 9))

plt.scatter(
    top_n["centroid_x"],
    top_n["centroid_y"],
    s=25,
    alpha=0.8
)

plt.gca().invert_yaxis()

plt.xlabel("X pixel")
plt.ylabel("Y pixel")

plt.title("Top 100 Candidate Locations — 15σ")

plt.tight_layout()

plot_file = os.path.join(
    VIS_DIR,
    "top_100_candidate_locations.png"
)

plt.savefig(
    plot_file,
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# 15. Final
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RANKING COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(f"  {ranked_file}")
print(f"  {top50_file}")

print("\nVisualizations:")
print(f"  {VIS_DIR}\\candidate_score_distribution.png")
print(f"  {VIS_DIR}\\candidate_score_vs_integrated.png")
print(f"  {VIS_DIR}\\candidate_score_vs_peak.png")
print(f"  {VIS_DIR}\\top_100_candidate_locations.png")