from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DIGANTARA VISUAL QA - STREAK ANNOTATIONS
# ============================================================

BASE_DIR = Path("output")

# Final annotation masks
MASK_DIR = BASE_DIR / "final_annotation" / "tiles" / "masks"

# QA output
QA_DIR = BASE_DIR / "QA" / "streak_review"
QA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. Find all mask files
# ------------------------------------------------------------

mask_files = sorted(MASK_DIR.glob("*.npy"))

print("=" * 70)
print("DIGANTARA VISUAL QA")
print("=" * 70)

print(f"Mask directory: {MASK_DIR}")
print(f"Total masks: {len(mask_files)}")


# ------------------------------------------------------------
# 2. Automatically find the folder containing image tiles
# ------------------------------------------------------------

mask_names = {f.name for f in mask_files}

candidate_dirs = []

for folder in BASE_DIR.rglob("*"):
    if folder.is_dir():

        npy_files = list(folder.glob("*.npy"))

        if not npy_files:
            continue

        # Don't use the final annotation mask folder
        if MASK_DIR.resolve() == folder.resolve():
            continue

        matching = sum(f.name in mask_names for f in npy_files)

        if matching > 0:
            candidate_dirs.append(
                (matching, folder)
            )


if not candidate_dirs:

    print()
    print("ERROR: Could not find the image-tile folder.")
    print()
    print("Run this command to locate your tiles:")
    print(
        'Get-ChildItem -Path output -Recurse -Filter "*.npy" '
        '| Where-Object { $_.FullName -notlike "*final_annotation*" } '
        '| Select-Object FullName'
    )

    raise SystemExit(1)


# Choose directory with most matching files
candidate_dirs.sort(reverse=True)

IMAGE_DIR = candidate_dirs[0][1]
matching_count = candidate_dirs[0][0]

print()
print("Image tile directory found:")
print(IMAGE_DIR)
print(f"Matching tiles: {matching_count}")


# ------------------------------------------------------------
# 3. Find masks containing Streak class = 2
# ------------------------------------------------------------

streak_masks = []

for mask_file in mask_files:

    mask = np.load(mask_file)

    if np.any(mask == 2):
        streak_masks.append(mask_file)


print()
print(f"Streak-containing masks: {len(streak_masks)}")
print()


# ------------------------------------------------------------
# 4. Create visual overlays
# ------------------------------------------------------------

successful = 0
missing = 0

for i, mask_file in enumerate(streak_masks, 1):

    name = mask_file.stem

    image_file = IMAGE_DIR / f"{name}.npy"

    if not image_file.exists():

        print(f"Missing image: {image_file}")
        missing += 1
        continue


    # Load data
    image = np.load(image_file)
    mask = np.load(mask_file)


    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if image.shape != (1024, 1024):

        print(
            f"WARNING: Image shape {image.shape}: "
            f"{image_file.name}"
        )


    if mask.shape != (1024, 1024):

        print(
            f"WARNING: Mask shape {mask.shape}: "
            f"{mask_file.name}"
        )


    # --------------------------------------------------------
    # Display stretch
    # --------------------------------------------------------

    low, high = np.percentile(
        image,
        [1, 99.8]
    )

    if high <= low:
        high = low + 1


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )


    # Original tile
    ax.imshow(
        image,
        cmap="gray",
        vmin=low,
        vmax=high,
        interpolation="nearest"
    )


    # --------------------------------------------------------
    # Blob class = 1
    # --------------------------------------------------------

    blob = np.ma.masked_where(
        mask != 1,
        mask
    )

    ax.imshow(
        blob,
        cmap="Blues",
        alpha=0.35,
        interpolation="nearest"
    )


    # --------------------------------------------------------
    # Streak class = 2
    # --------------------------------------------------------

    streak = np.ma.masked_where(
        mask != 2,
        mask
    )

    ax.imshow(
        streak,
        cmap="Reds",
        alpha=0.9,
        interpolation="nearest"
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    blob_pixels = int(
        np.sum(mask == 1)
    )

    streak_pixels = int(
        np.sum(mask == 2)
    )


    ax.set_title(
        f"{i}/{len(streak_masks)}\n"
        f"{name}\n"
        f"Blob pixels: {blob_pixels} | "
        f"Streak pixels: {streak_pixels}",
        fontsize=10
    )

    ax.axis("off")

    plt.tight_layout()


    # --------------------------------------------------------
    # Save QA image
    # --------------------------------------------------------

    output_file = (
        QA_DIR /
        f"{i:03d}_{name}.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    successful += 1


# ------------------------------------------------------------
# 5. Final report
# ------------------------------------------------------------

print()
print("=" * 70)
print("VISUAL QA COMPLETE")
print("=" * 70)

print(f"Total masks:              {len(mask_files)}")
print(f"Streak masks:             {len(streak_masks)}")
print(f"QA images created:        {successful}")
print(f"Missing image tiles:      {missing}")

print()
print("QA images saved to:")
print(QA_DIR)

print("=" * 70)