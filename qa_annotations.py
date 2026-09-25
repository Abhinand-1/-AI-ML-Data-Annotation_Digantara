"""
Digantara Annotation QA

Validates final 1024x1024 NumPy annotation masks.

Classes:
    0 = Background
    1 = Blob / Star
    2 = Streak / Space Object
"""

from pathlib import Path
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MASK_DIR = (
    ROOT
    / "output"
    / "final_annotation"
    / "tiles"
    / "masks"
)

TILE_SIZE = 1024

VALID_CLASSES = {
    0: "Background",
    1: "Blob / Star",
    2: "Streak / Space Object",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ANNOTATION QA")
print("=" * 70)

print(f"Project root : {ROOT}")
print(f"Mask directory : {MASK_DIR}")


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not MASK_DIR.exists():
    print()
    print("ERROR: Mask directory does not exist.")
    print(f"Expected: {MASK_DIR}")
    raise SystemExit(1)


# ============================================================
# FIND MASKS
# ============================================================

mask_files = sorted(
    MASK_DIR.glob("*.npy")
)

print(f"Masks found    : {len(mask_files)}")


if not mask_files:
    print()
    print("ERROR: No .npy masks found.")
    print("Check the annotation application and MASK_DIR.")
    raise SystemExit(1)


# ============================================================
# QA
# ============================================================

issues = []
passed = 0

total_blob_pixels = 0
total_streak_pixels = 0
total_background_pixels = 0


for mask_path in mask_files:

    file_issues = []

    try:
        mask = np.load(mask_path)

    except Exception as e:
        file_issues.append(
            f"Could not load mask: {e}"
        )

        issues.append(
            (mask_path.name, file_issues)
        )

        continue


    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    if mask.shape != (TILE_SIZE, TILE_SIZE):

        file_issues.append(
            f"Invalid shape: {mask.shape}"
        )


    # --------------------------------------------------------
    # Dtype
    # --------------------------------------------------------

    if mask.dtype != np.uint8:

        file_issues.append(
            f"Invalid dtype: {mask.dtype}; expected uint8"
        )


    # --------------------------------------------------------
    # Class values
    # --------------------------------------------------------

    unique_values = np.unique(mask)

    invalid_values = [
        int(v)
        for v in unique_values
        if int(v) not in VALID_CLASSES
    ]

    if invalid_values:

        file_issues.append(
            f"Invalid class values: {invalid_values}"
        )


    # --------------------------------------------------------
    # Pixel counts
    # --------------------------------------------------------

    background_pixels = int(
        np.sum(mask == 0)
    )

    blob_pixels = int(
        np.sum(mask == 1)
    )

    streak_pixels = int(
        np.sum(mask == 2)
    )

    total_background_pixels += background_pixels
    total_blob_pixels += blob_pixels
    total_streak_pixels += streak_pixels


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if file_issues:

        issues.append(
            (mask_path.name, file_issues)
        )

    else:

        passed += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("QA SUMMARY")
print("=" * 70)

print(f"Total masks       : {len(mask_files)}")
print(f"Passed            : {passed}")
print(f"Failed            : {len(issues)}")

print()
print("Pixel statistics")
print("-" * 70)

print(
    f"Background pixels : {total_background_pixels:,}"
)

print(
    f"Blob pixels       : {total_blob_pixels:,}"
)

print(
    f"Streak pixels     : {total_streak_pixels:,}"
)


# ============================================================
# FAILED FILES
# ============================================================

if issues:

    print()
    print("=" * 70)
    print("FAILED MASKS")
    print("=" * 70)

    for filename, file_issues in issues:

        print()
        print(f"{filename} ->")

        for issue in file_issues:

            print(f"   - {issue}")

else:

    print()
    print("=" * 70)
    print("ALL MASKS PASSED AUTOMATED QA")
    print("=" * 70)


print()