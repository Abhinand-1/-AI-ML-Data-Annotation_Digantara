from pathlib import Path
import numpy as np
import cv2
import shutil


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MASK_DIR = ROOT / "output" / "final_annotation" / "tiles" / "masks"
IMAGE_DIR = ROOT / "output" / "final_annotation" / "tiles" / "images"

YOLO_DIR = ROOT / "output" / "yolo_dataset"

YOLO_IMAGES = YOLO_DIR / "images"
YOLO_LABELS = YOLO_DIR / "labels"

YOLO_IMAGES.mkdir(parents=True, exist_ok=True)
YOLO_LABELS.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 1024

# Internal mask classes:
# 0 = background
# 1 = blob
# 2 = streak
#
# YOLO classes:
# 0 = blob
# 1 = streak

MASK_TO_YOLO = {
    1: 0,
    2: 1,
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DIGANTARA MASK → YOLO SEGMENTATION CONVERSION")
print("=" * 70)

print(f"Mask directory  : {MASK_DIR}")
print(f"Image directory : {IMAGE_DIR}")
print(f"YOLO directory  : {YOLO_DIR}")


# ============================================================
# CHECK INPUT
# ============================================================

if not MASK_DIR.exists():
    raise SystemExit(
        f"\nERROR: Mask directory not found:\n{MASK_DIR}"
    )

if not IMAGE_DIR.exists():
    raise SystemExit(
        f"\nERROR: Image directory not found:\n{IMAGE_DIR}"
    )


mask_files = sorted(MASK_DIR.glob("*.npy"))

print(f"\nMasks found: {len(mask_files)}")

if not mask_files:
    raise SystemExit("ERROR: No .npy masks found.")


# ============================================================
# CONVERT ONE CLASS TO POLYGONS
# ============================================================

def mask_to_polygons(binary_mask):

    mask = (binary_mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    for contour in contours:

        # Ignore extremely tiny components
        if cv2.contourArea(contour) < 1:
            continue

        epsilon = 0.002 * cv2.arcLength(
            contour,
            True
        )

        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        if len(contour) < 3:
            continue

        points = contour.reshape(-1, 2)

        normalized = []

        for x, y in points:

            xn = float(x) / IMAGE_SIZE
            yn = float(y) / IMAGE_SIZE

            xn = min(max(xn, 0.0), 1.0)
            yn = min(max(yn, 0.0), 1.0)

            normalized.extend([xn, yn])

        if len(normalized) >= 6:
            polygons.append(normalized)

    return polygons


# ============================================================
# CONVERSION
# ============================================================

total_labels = 0
blob_objects = 0
streak_objects = 0
empty_masks = 0
failed = 0


for i, mask_path in enumerate(mask_files, 1):

    try:

        mask = np.load(mask_path)

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if mask.shape != (1024, 1024):
            print(
                f"[SKIP] {mask_path.name}: "
                f"shape={mask.shape}"
            )
            failed += 1
            continue

        # ----------------------------------------------------
        # YOLO label file
        # ----------------------------------------------------

        label_path = YOLO_LABELS / (
            mask_path.stem + ".txt"
        )

        lines = []

        # ----------------------------------------------------
        # Blob
        # ----------------------------------------------------

        blob_mask = (mask == 1).astype(np.uint8)

        blob_polygons = mask_to_polygons(
            blob_mask
        )

        for polygon in blob_polygons:

            line = "0 " + " ".join(
                f"{v:.6f}" for v in polygon
            )

            lines.append(line)
            blob_objects += 1
            total_labels += 1

        # ----------------------------------------------------
        # Streak
        # ----------------------------------------------------

        streak_mask = (mask == 2).astype(np.uint8)

        streak_polygons = mask_to_polygons(
            streak_mask
        )

        for polygon in streak_polygons:

            line = "1 " + " ".join(
                f"{v:.6f}" for v in polygon
            )

            lines.append(line)
            streak_objects += 1
            total_labels += 1

        # ----------------------------------------------------
        # Save label
        # ----------------------------------------------------

        label_path.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        if not lines:
            empty_masks += 1

        # ----------------------------------------------------
        # Copy corresponding image
        # ----------------------------------------------------

        image_path = IMAGE_DIR / (
            mask_path.stem + ".npy"
        )

        if image_path.exists():

            shutil.copy2(
                image_path,
                YOLO_IMAGES / image_path.name
            )

        else:

            print(
                f"[WARNING] Image missing: "
                f"{image_path.name}"
            )

        if i % 25 == 0 or i == len(mask_files):

            print(
                f"Processed {i}/{len(mask_files)}"
            )

    except Exception as e:

        failed += 1

        print(
            f"[ERROR] {mask_path.name}: {e}"
        )


# ============================================================
# CREATE DATASET YAML
# ============================================================

yaml_text = """path: .

train: images
val: images

names:
  0: blob
  1: streak
"""

(YOLO_DIR / "data.yaml").write_text(
    yaml_text,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)

print(f"Input masks       : {len(mask_files)}")
print(f"YOLO objects      : {total_labels}")
print(f"Blob objects      : {blob_objects}")
print(f"Streak objects    : {streak_objects}")
print(f"Empty labels      : {empty_masks}")
print(f"Failed masks      : {failed}")

print()
print("Output:")
print(f"Images : {YOLO_IMAGES}")
print(f"Labels : {YOLO_LABELS}")
print(f"YAML   : {YOLO_DIR / 'data.yaml'}")