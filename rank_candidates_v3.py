"""
DIGANTARA — V3 Candidate Ranking

Purpose:
    Rank candidate detections from all FITS images using
    within-image normalized signal + morphology.

Input:
    output/all_fits_candidates.csv

Outputs:
    output/v3/all_fits_candidates_ranked_v3.csv
    output/v3/top_10_per_file_v3.csv
    output/v3/top_100_overall_v3.csv
    output/v3/v3_ranking_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Abhinand\DIGATRA_CODE_FILES")

INPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "all_fits_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "v3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD CANDIDATE DATA
# ============================================================

print("=" * 75)
print("DIGANTARA — V3 CANDIDATE RANKING")
print("=" * 75)

print("\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}\n\n"
        "Make sure process_all_fits.py has been run first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nCandidates loaded: {len(df):,}")

if "file" not in df.columns:
    raise ValueError(
        "Column 'file' is missing from the candidate CSV."
    )


# ============================================================
# 3. REQUIRED COLUMN CHECK
# ============================================================

required_columns = [
    "file",
    "component_id",
    "area",
    "width",
    "height",
    "aspect_ratio",
    "peak_residual",
    "mean_residual",
    "integrated_residual",
    "compactness",
    "robust_sigma",
    "centroid_x",
    "centroid_y",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(f"  - {c}" for c in missing_columns)
    )


# ============================================================
# 4. PREPARE NUMERIC VALUES
# ============================================================

numeric_columns = [
    "area",
    "width",
    "height",
    "aspect_ratio",
    "peak_residual",
    "mean_residual",
    "integrated_residual",
    "compactness",
    "robust_sigma",
    "centroid_x",
    "centroid_y",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# Remove rows with unusable core measurements.

df = df.dropna(
    subset=[
        "file",
        "area",
        "peak_residual",
        "integrated_residual",
        "robust_sigma",
    ]
).copy()

print(f"Valid candidates: {len(df):,}")


# ============================================================
# 5. HANDLE ZERO / INVALID SIGMA
# ============================================================

# Robust sigma should normally be > 0.
# We prevent division-by-zero from creating infinite scores.

df["safe_sigma"] = df["robust_sigma"].replace(
    0,
    np.nan
)

df["safe_sigma"] = df["safe_sigma"].abs()

# If any invalid values remain, use a tiny positive value.
df["safe_sigma"] = df["safe_sigma"].fillna(
    1e-12
)


# ============================================================
# 6. WITHIN-FILE SIGNAL NORMALIZATION
# ============================================================

"""
The FITS files have different background/noise scales.

Therefore, raw peak intensity should not be compared directly
between different FITS files.

Instead we calculate signal relative to each image's robust sigma.
"""

df["peak_sigma"] = (
    df["peak_residual"]
    / df["safe_sigma"]
)

df["mean_sigma"] = (
    df["mean_residual"]
    / df["safe_sigma"]
)

# Integrated signal is normalized by sqrt(area) so that
# larger components do not automatically dominate.

df["integrated_sigma"] = (
    df["integrated_residual"]
    /
    (
        np.sqrt(
            df["area"].clip(lower=1)
        )
        * df["safe_sigma"]
    )
)


# ============================================================
# 7. PERCENTILE-RANK FUNCTION
# ============================================================

def percentile_rank(series):
    """
    Convert values into percentile ranks from 0 to 1.
    """
    return series.rank(
        method="average",
        pct=True
    )


# ============================================================
# 8. NORMALIZED SIGNAL SCORES
# ============================================================

grouped = df.groupby(
    "file",
    group_keys=False
)

# Signal strength within each image.

df["peak_score"] = grouped[
    "peak_sigma"
].transform(percentile_rank)

df["integrated_score"] = grouped[
    "integrated_sigma"
].transform(percentile_rank)

df["mean_score"] = grouped[
    "mean_sigma"
].transform(percentile_rank)


# ============================================================
# 9. MORPHOLOGY SCORES
# ============================================================

# Larger connected components receive a moderate score.

df["area_score"] = grouped[
    "area"
].transform(percentile_rank)


# Compactness is already calculated during candidate extraction.

df["compactness_score"] = grouped[
    "compactness"
].transform(percentile_rank)


# ============================================================
# 10. SHAPE SCORE
# ============================================================

"""
A compact approximately symmetric candidate has an aspect ratio
near 1.

This does NOT mean every real object must have aspect ratio 1.
It is simply one morphology feature used in the ranking.
"""

df["shape_score"] = np.exp(
    -(
        (
            df["aspect_ratio"] - 1.0
        )
        / 0.35
    ) ** 2
)


# ============================================================
# 11. V3 COMPOSITE SCORE
# ============================================================

"""
Weights:

    Peak signal       35%
    Integrated signal 25%
    Mean signal       10%
    Area              10%
    Compactness       10%
    Shape             10%

This is a heuristic prioritization score.

It is NOT a trained classifier and does not prove
that a candidate is a satellite/object.
"""

df["candidate_score_v3"] = (
    0.35 * df["peak_score"]
    +
    0.25 * df["integrated_score"]
    +
    0.10 * df["mean_score"]
    +
    0.10 * df["area_score"]
    +
    0.10 * df["compactness_score"]
    +
    0.10 * df["shape_score"]
)


# ============================================================
# 12. RANK WITHIN EACH FITS IMAGE
# ============================================================

df["rank_within_file"] = (
    df.groupby("file")[
        "candidate_score_v3"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


# Percentile position within each image.

df["percentile_within_file"] = (
    df.groupby("file")[
        "candidate_score_v3"
    ]
    .rank(
        method="average",
        pct=True
    )
)


# ============================================================
# 13. PRIORITY LEVEL
# ============================================================

"""
Priority 1:
    Top 5% within each image.

Priority 2:
    80th–95th percentile.

Priority 3:
    50th–80th percentile.

Low:
    Bottom 50%.
"""

df["priority"] = np.select(
    [
        df["percentile_within_file"] >= 0.95,

        df["percentile_within_file"] >= 0.80,

        df["percentile_within_file"] >= 0.50,
    ],
    [
        "Priority_1",
        "Priority_2",
        "Priority_3",
    ],
    default="Low"
)


# ============================================================
# 14. SORT RESULTS
# ============================================================

df = df.sort_values(
    [
        "file",
        "rank_within_file"
    ]
).reset_index(drop=True)


# ============================================================
# 15. SAVE COMPLETE RANKING
# ============================================================

ranked_file = (
    OUTPUT_DIR
    / "all_fits_candidates_ranked_v3.csv"
)

df.to_csv(
    ranked_file,
    index=False
)


# ============================================================
# 16. TOP 10 FROM EACH FITS
# ============================================================

top_10_per_file = df[
    df["rank_within_file"] <= 10
].copy()

top_10_file = (
    OUTPUT_DIR
    / "top_10_per_file_v3.csv"
)

top_10_per_file.to_csv(
    top_10_file,
    index=False
)


# ============================================================
# 17. TOP 100 OVERALL
# ============================================================

top_100_overall = (
    df
    .sort_values(
        "candidate_score_v3",
        ascending=False
    )
    .head(100)
    .copy()
)

top_100_file = (
    OUTPUT_DIR
    / "top_100_overall_v3.csv"
)

top_100_overall.to_csv(
    top_100_file,
    index=False
)


# ============================================================
# 18. PER-FILE SUMMARY
# ============================================================

summary = (
    df
    .groupby("file")
    .agg(
        candidates=(
            "component_id",
            "size"
        ),

        priority_1=(
            "priority",
            lambda x:
            (x == "Priority_1").sum()
        ),

        priority_2=(
            "priority",
            lambda x:
            (x == "Priority_2").sum()
        ),

        priority_3=(
            "priority",
            lambda x:
            (x == "Priority_3").sum()
        ),

        low=(
            "priority",
            lambda x:
            (x == "Low").sum()
        ),

        top_score=(
            "candidate_score_v3",
            "max"
        ),

        median_score=(
            "candidate_score_v3",
            "median"
        ),
    )
    .reset_index()
)

summary_file = (
    OUTPUT_DIR
    / "v3_ranking_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 19. TERMINAL REPORT
# ============================================================

print("\n")
print("=" * 75)
print("V3 RANKING COMPLETE")
print("=" * 75)

print(
    f"\nTotal candidates ranked: "
    f"{len(df):,}"
)

print("\nPriority counts:")
print(
    df["priority"]
    .value_counts()
    .to_string()
)

print("\nPer-file summary:")
print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 20. TOP 20 OVERALL
# ============================================================

display_columns = [
    "file",
    "component_id",
    "rank_within_file",
    "candidate_score_v3",
    "peak_residual",
    "integrated_residual",
    "area",
    "compactness",
    "aspect_ratio",
    "centroid_x",
    "centroid_y",
]

print("\n")
print("=" * 75)
print("TOP 20 CANDIDATES OVERALL")
print("=" * 75)

print(
    top_100_overall[
        display_columns
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 21. OUTPUT FILES
# ============================================================

print("\n")
print("=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print("\nComplete ranking:")
print(ranked_file)

print("\nTop 10 per FITS:")
print(top_10_file)

print("\nTop 100 overall:")
print(top_100_file)

print("\nV3 summary:")
print(summary_file)

print("\n")
print("=" * 75)
print("V3 CANDIDATE RANKING FINISHED")
print("=" * 75)