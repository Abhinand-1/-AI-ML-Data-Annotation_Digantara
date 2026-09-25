from pathlib import Path
from collections import Counter
import csv


# ============================================================
# DIGANTARA YOLO DATASET VALIDATION
# ============================================================

BASE_DIR = Path(r"C:\Users\Abhinand\DIGATRA_CODE_FILES")

IMAGE_DIR = BASE_DIR / "output" / "yolo_dataset" / "images"
LABEL_DIR = BASE_DIR / "output" / "yolo_dataset" / "labels"

REPORT_DIR = BASE_DIR / "output" / "validation_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = REPORT_DIR / "annotation_statistics.csv"
TXT_FILE = REPORT_DIR / "dataset_validation_report.txt"


# ============================================================
# IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
}


# ============================================================
# CHECK DIRECTORIES
# ============================================================

print("=" * 70)
print("DIGANTARA YOLO DATASET VALIDATION")
print("=" * 70)

print("\nChecking directories...")

if not IMAGE_DIR.exists():
    print("\nERROR: Image directory does not exist:")
    print(IMAGE_DIR)
    input("\nPress Enter to exit...")
    raise SystemExit

if not LABEL_DIR.exists():
    print("\nERROR: Label directory does not exist:")
    print(LABEL_DIR)
    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# FIND IMAGES AND LABELS
# ============================================================

images = sorted(
    [
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)

labels = sorted(
    [
        p
        for p in LABEL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]
)


print("\nImage directory:")
print(IMAGE_DIR)

print("\nLabel directory:")
print(LABEL_DIR)

print("\nImages found :", len(images))
print("Labels found :", len(labels))


# ============================================================
# IMAGE / LABEL MATCHING
# ============================================================

image_names = {p.stem for p in images}
label_names = {p.stem for p in labels}

missing_labels = sorted(image_names - label_names)
orphan_labels = sorted(label_names - image_names)


print("\n" + "-" * 70)
print("IMAGE / LABEL MATCHING")
print("-" * 70)

print("Images without labels :", len(missing_labels))
print("Labels without images :", len(orphan_labels))


if missing_labels:

    print("\nImages without labels:")

    for name in missing_labels[:20]:
        print("  ", name)


if orphan_labels:

    print("\nLabels without images:")

    for name in orphan_labels[:20]:
        print("  ", name)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

# Expected YOLO classes:
#
# 0 = blob
# 1 = streak

CLASS_NAMES = {
    0: "blob",
    1: "streak"
}


# ============================================================
# VARIABLES
# ============================================================

class_counter = Counter()

image_annotation_counts = []

invalid_lines = []
invalid_coordinates = []

total_annotations = 0

empty_label_files = 0
nonempty_label_files = 0


# ============================================================
# READ LABEL FILES
# ============================================================

print("\n" + "-" * 70)
print("READING YOLO LABELS")
print("-" * 70)


for label_file in labels:

    annotation_count = 0
    blob_count = 0
    streak_count = 0

    try:

        with open(
            label_file,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

    except Exception as e:

        invalid_lines.append(
            (
                label_file.name,
                f"Could not read file: {e}"
            )
        )

        continue


    # Empty label file
    if len(lines) == 0:

        empty_label_files += 1

        image_annotation_counts.append(
            (
                label_file.stem,
                0,
                0,
                0
            )
        )

        continue


    nonempty_label_files += 1


    # --------------------------------------------------------
    # READ EACH ANNOTATION
    # --------------------------------------------------------

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        parts = line.split()


        # YOLO segmentation needs:
        #
        # class
        # x1 y1
        # x2 y2
        # x3 y3
        # ...
        #
        # Minimum:
        # class + 3 points = 7 values

        if len(parts) < 7:

            invalid_lines.append(
                (
                    label_file.name,
                    f"Line {line_number}: "
                    f"insufficient polygon points"
                )
            )

            continue


        # ----------------------------------------------------
        # CLASS ID
        # ----------------------------------------------------

        try:

            class_id = int(parts[0])

        except ValueError:

            invalid_lines.append(
                (
                    label_file.name,
                    f"Line {line_number}: "
                    f"invalid class ID"
                )
            )

            continue


        # ----------------------------------------------------
        # CLASS VALIDATION
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:

            invalid_lines.append(
                (
                    label_file.name,
                    f"Line {line_number}: "
                    f"unknown class {class_id}"
                )
            )

            continue


        # ----------------------------------------------------
        # COORDINATES
        # ----------------------------------------------------

        try:

            coordinates = [
                float(x)
                for x in parts[1:]
            ]

        except ValueError:

            invalid_lines.append(
                (
                    label_file.name,
                    f"Line {line_number}: "
                    f"invalid coordinates"
                )
            )

            continue


        # ----------------------------------------------------
        # EVEN NUMBER OF COORDINATES
        # ----------------------------------------------------

        if len(coordinates) % 2 != 0:

            invalid_lines.append(
                (
                    label_file.name,
                    f"Line {line_number}: "
                    f"odd number of coordinates"
                )
            )

            continue


        # ----------------------------------------------------
        # NORMALIZED COORDINATES
        # ----------------------------------------------------

        bad_coordinate = False

        for value in coordinates:

            if value < 0 or value > 1:

                bad_coordinate = True
                break


        if bad_coordinate:

            invalid_coordinates.append(
                (
                    label_file.name,
                    f"Line {line_number}: "
                    f"coordinate outside [0,1]"
                )
            )


        # ----------------------------------------------------
        # COUNT ANNOTATION
        # ----------------------------------------------------

        annotation_count += 1

        total_annotations += 1

        class_counter[class_id] += 1


        if class_id == 0:

            blob_count += 1

        elif class_id == 1:

            streak_count += 1


    # --------------------------------------------------------
    # SAVE PER IMAGE STATISTICS
    # --------------------------------------------------------

    image_annotation_counts.append(
        (
            label_file.stem,
            annotation_count,
            blob_count,
            streak_count
        )
    )


# ============================================================
# STATISTICS
# ============================================================

total_blobs = class_counter[0]
total_streaks = class_counter[1]


annotated_images = sum(
    1
    for _, total, _, _
    in image_annotation_counts
    if total > 0
)


empty_images = sum(
    1
    for _, total, _, _
    in image_annotation_counts
    if total == 0
)


if labels:

    average_annotations = (
        total_annotations / len(labels)
    )

else:

    average_annotations = 0


max_annotations = max(
    (
        x[1]
        for x in image_annotation_counts
    ),
    default=0
)


min_annotations = min(
    (
        x[1]
        for x in image_annotation_counts
    ),
    default=0
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANNOTATION SUMMARY")
print("=" * 70)

print("\nTotal images             :", len(images))
print("Total label files        :", len(labels))

print("\nAnnotated images         :", annotated_images)
print("Empty/background images  :", empty_images)

print("\nTotal annotations        :", total_annotations)
print("Blob annotations         :", total_blobs)
print("Streak annotations       :", total_streaks)

print("\nAverage annotations/file :",
      f"{average_annotations:.2f}")

print("Minimum annotations/file:",
      min_annotations)

print("Maximum annotations/file:",
      max_annotations)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("CLASS DISTRIBUTION")
print("-" * 70)


for class_id, class_name in CLASS_NAMES.items():

    count = class_counter[class_id]

    if total_annotations > 0:

        percentage = (
            count /
            total_annotations *
            100
        )

    else:

        percentage = 0


    print(
        f"{class_id} = {class_name:<10} "
        f"{count:>8} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("VALIDATION RESULTS")
print("-" * 70)

print(
    "Invalid label lines      :",
    len(invalid_lines)
)

print(
    "Invalid coordinates      :",
    len(invalid_coordinates)
)


if len(invalid_lines) == 0:

    print("Label syntax             : OK")

else:

    print("Label syntax             : CHECK REQUIRED")


if len(invalid_coordinates) == 0:

    print("Coordinate normalization : OK")

else:

    print("Coordinate normalization : CHECK REQUIRED")


if len(missing_labels) == 0:

    print("Missing labels           : NONE")

else:

    print("Missing labels           : CHECK REQUIRED")


if len(orphan_labels) == 0:

    print("Orphan labels            : NONE")

else:

    print("Orphan labels            : CHECK REQUIRED")


# ============================================================
# WRITE CSV
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "image_name",
            "total_annotations",
            "blob_annotations",
            "streak_annotations"
        ]
    )


    for row in image_annotation_counts:

        writer.writerow(row)


# ============================================================
# WRITE TEXT REPORT
# ============================================================

with open(
    TXT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 70 +
        "\n"
    )

    f.write(
        "DIGANTARA DATASET VALIDATION REPORT\n"
    )

    f.write(
        "=" * 70 +
        "\n\n"
    )


    f.write(
        "DATASET\n"
    )

    f.write(
        "-" * 70 +
        "\n"
    )

    f.write(
        f"Image directory : {IMAGE_DIR}\n"
    )

    f.write(
        f"Label directory : {LABEL_DIR}\n"
    )

    f.write(
        f"Total images    : {len(images)}\n"
    )

    f.write(
        f"Total labels    : {len(labels)}\n\n"
    )


    f.write(
        "ANNOTATION SUMMARY\n"
    )

    f.write(
        "-" * 70 +
        "\n"
    )

    f.write(
        f"Annotated images        : "
        f"{annotated_images}\n"
    )

    f.write(
        f"Empty/background images : "
        f"{empty_images}\n"
    )

    f.write(
        f"Total annotations       : "
        f"{total_annotations}\n"
    )

    f.write(
        f"Blob annotations        : "
        f"{total_blobs}\n"
    )

    f.write(
        f"Streak annotations      : "
        f"{total_streaks}\n"
    )

    f.write(
        f"Average annotations     : "
        f"{average_annotations:.2f}\n"
    )

    f.write(
        f"Minimum annotations     : "
        f"{min_annotations}\n"
    )

    f.write(
        f"Maximum annotations     : "
        f"{max_annotations}\n\n"
    )


    f.write(
        "CLASS DISTRIBUTION\n"
    )

    f.write(
        "-" * 70 +
        "\n"
    )


    for class_id, class_name in CLASS_NAMES.items():

        count = class_counter[class_id]

        percentage = (
            count /
            total_annotations *
            100
            if total_annotations > 0
            else 0
        )

        f.write(
            f"{class_id} = {class_name}: "
            f"{count} "
            f"({percentage:.2f}%)\n"
        )


    f.write("\n")

    f.write(
        "VALIDATION\n"
    )

    f.write(
        "-" * 70 +
        "\n"
    )

    f.write(
        f"Missing labels      : "
        f"{len(missing_labels)}\n"
    )

    f.write(
        f"Orphan labels       : "
        f"{len(orphan_labels)}\n"
    )

    f.write(
        f"Invalid label lines : "
        f"{len(invalid_lines)}\n"
    )

    f.write(
        f"Invalid coordinates : "
        f"{len(invalid_coordinates)}\n"
    )


    # --------------------------------------------------------
    # INVALID LABEL DETAILS
    # --------------------------------------------------------

    if invalid_lines:

        f.write("\n\n")

        f.write(
            "INVALID LABEL LINES\n"
        )

        f.write(
            "-" * 70 +
            "\n"
        )


        for filename, message in invalid_lines:

            f.write(
                f"{filename}: {message}\n"
            )


    # --------------------------------------------------------
    # INVALID COORDINATES
    # --------------------------------------------------------

    if invalid_coordinates:

        f.write("\n\n")

        f.write(
            "INVALID COORDINATES\n"
        )

        f.write(
            "-" * 70 +
            "\n"
        )


        for filename, message in invalid_coordinates:

            f.write(
                f"{filename}: {message}\n"
            )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print("\nReports created:")

print("\nCSV:")
print(CSV_FILE)

print("\nTXT:")
print(TXT_FILE)

print("\nThe dataset was NOT modified.")

print("\nNext step:")
print(
    "Send me the terminal output and I will check "
    "whether the YOLO dataset is ready for final QA "
    "and the Digantara report."
)

input("\nPress Enter to exit...")