from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "output" / "yolo_dataset" / "images"
LABEL_DIR = ROOT / "output" / "yolo_dataset" / "labels"


print("=" * 70)
print("FINAL YOLO DATASET QA")
print("=" * 70)

images = {
    p.stem
    for p in IMAGE_DIR.glob("*.png")
}

labels = {
    p.stem
    for p in LABEL_DIR.glob("*.txt")
}

print(f"PNG images : {len(images)}")
print(f"Labels     : {len(labels)}")


# ------------------------------------------------------------
# Matching
# ------------------------------------------------------------

missing_images = sorted(labels - images)
missing_labels = sorted(images - labels)

print()
print("MATCHING")
print("-" * 70)

print(f"Labels without images : {len(missing_images)}")
print(f"Images without labels : {len(missing_labels)}")


# ------------------------------------------------------------
# Empty labels
# ------------------------------------------------------------

empty_labels = []

for path in LABEL_DIR.glob("*.txt"):

    if path.stat().st_size == 0:
        empty_labels.append(path.name)

print()
print("EMPTY LABELS")
print("-" * 70)

print(f"Empty labels : {len(empty_labels)}")

for name in empty_labels:
    print("  ", name)


# ------------------------------------------------------------
# Polygon validation
# ------------------------------------------------------------

bad_format = []
bad_class = []
bad_coordinates = []

polygon_count = 0

for path in LABEL_DIR.glob("*.txt"):

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, line in enumerate(lines, 1):

        if not line.strip():
            continue

        parts = line.split()

        # class + x/y pairs
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            bad_format.append(
                (path.name, line_number, line)
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            bad_format.append(
                (path.name, line_number, line)
            )
            continue

        if class_id not in (0, 1):
            bad_class.append(
                (path.name, line_number, class_id)
            )

        try:
            coords = [
                float(x)
                for x in parts[1:]
            ]
        except ValueError:
            bad_format.append(
                (path.name, line_number, line)
            )
            continue

        if any(
            x < 0 or x > 1
            for x in coords
        ):
            bad_coordinates.append(
                (path.name, line_number, line)
            )

        polygon_count += 1


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("QA SUMMARY")
print("=" * 70)

print(f"PNG images           : {len(images)}")
print(f"YOLO labels          : {len(labels)}")
print(f"YOLO polygons        : {polygon_count}")
print(f"Empty labels         : {len(empty_labels)}")
print(f"Missing images       : {len(missing_images)}")
print(f"Missing labels       : {len(missing_labels)}")
print(f"Bad polygon format   : {len(bad_format)}")
print(f"Bad class IDs        : {len(bad_class)}")
print(f"Bad coordinates      : {len(bad_coordinates)}")


# ------------------------------------------------------------
# Final status
# ------------------------------------------------------------

if (
    len(missing_images) == 0
    and len(missing_labels) == 0
    and len(bad_format) == 0
    and len(bad_class) == 0
    and len(bad_coordinates) == 0
):

    print()
    print("=" * 70)
    print("YOLO DATASET STRUCTURE PASSED")
    print("=" * 70)

else:

    print()
    print("=" * 70)
    print("YOLO DATASET HAS ISSUES")
    print("=" * 70)