from pathlib import Path

import numpy as np
from astropy.io import fits


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

fits_files = sorted(RAW_DIR.glob("*.fits"))

print("=" * 80)
print("DIGANTARA FITS PIXEL ENCODING ANALYSIS")
print("=" * 80)


for i, fits_path in enumerate(fits_files, start=1):

    print("\n" + "-" * 80)
    print(f"[{i}] {fits_path.name}")

    with fits.open(fits_path, memmap=False) as hdul:

        data = hdul[0].data
        header = hdul[0].header

        data = np.asarray(data, dtype=np.float32)
        data = np.squeeze(data)

    total = data.size

    print(f"\nShape       : {data.shape}")
    print(f"BSCALE      : {header.get('BSCALE')}")
    print(f"BZERO       : {header.get('BZERO')}")

    print("\nScaled pixel statistics:")
    print(f"Minimum     : {np.min(data):.3f}")
    print(f"Maximum     : {np.max(data):.3f}")
    print(f"Mean        : {np.mean(data):.3f}")
    print(f"Median      : {np.median(data):.3f}")

    # Important pixel counts
    zero_count = np.count_nonzero(data == 0)
    one_count = np.count_nonzero(data == 1)
    two_count = np.count_nonzero(data == 2)

    print("\nImportant pixel counts:")
    print(f"Pixels == 0 : {zero_count:,} ({zero_count / total * 100:.4f}%)")
    print(f"Pixels == 1 : {one_count:,} ({one_count / total * 100:.4f}%)")
    print(f"Pixels == 2 : {two_count:,} ({two_count / total * 100:.4f}%)")

    # High-value pixels
    for threshold in [10, 100, 1000, 4095, 65535]:

        count = np.count_nonzero(data >= threshold)

        print(
            f"Pixels >= {threshold:5d}: "
            f"{count:,} "
            f"({count / total * 100:.6f}%)"
        )


print("\n" + "=" * 80)
print("ENCODING ANALYSIS COMPLETE")
print("=" * 80)