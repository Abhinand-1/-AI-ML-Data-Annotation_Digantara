from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT / "output" / "final_annotation" / "tiles" / "images"
YOLO_DIR = ROOT / "output" / "yolo_dataset"
OUTPUT_DIR = YOLO_DIR / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(SOURCE_DIR.glob("*.npy"))

print("=" * 70)
print("NPY → PNG IMAGE CONVERSION")
print("=" * 70)
print("Input :", SOURCE_DIR)
print("Output:", OUTPUT_DIR)
print("Files :", len(files))

converted = 0
failed = 0

for i, path in enumerate(files, 1):

    try:
        arr = np.load(path)

        if arr.shape != (1024, 1024):
            print(f"[SKIP] {path.name}: shape={arr.shape}")
            failed += 1
            continue

        arr = arr.astype(np.float32)

        # Robust per-tile contrast scaling
        lo = np.percentile(arr, 1)
        hi = np.percentile(arr, 99.5)

        if hi <= lo:
            lo = arr.min()
            hi = arr.max()

        if hi <= lo:
            out = np.zeros((1024, 1024), dtype=np.uint8)
        else:
            out = (arr - lo) / (hi - lo)
            out = np.clip(out, 0, 1)
            out = (out * 255).astype(np.uint8)

        output_path = OUTPUT_DIR / f"{path.stem}.png"

        Image.fromarray(out, mode="L").save(
            output_path
        )

        converted += 1

        if i % 25 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)}")

    except Exception as e:
        print(f"[ERROR] {path.name}: {e}")
        failed += 1

print()
print("=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)
print("Input files :", len(files))
print("Converted   :", converted)
print("Failed      :", failed)
print("PNG output  :", OUTPUT_DIR)