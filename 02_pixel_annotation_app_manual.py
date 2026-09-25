"""
Digantara — Streamlit Manual Pixel-Level Blob / Streak Annotation

Classes
-------
0 = Background
1 = Blob / Star
2 = Streak / Space Object

This version is designed for fully manual annotation:
- Click-and-drag painting for blobs and streaks
- Eraser
- Undo / Redo
- Clear / Reset
- Zoom + pan controls
- Optional candidate boxes
- Optional existing-mask overlay
- Exact 1024 x 1024 uint8 .npy masks

Install:
    pip install streamlit streamlit-drawable-canvas opencv-python numpy pandas pillow

Run:
    streamlit run 02_pixel_annotation_app_streamlit.py
"""

from pathlib import Path
import csv
import re

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"C:\Users\Abhinand\DIGATRA_CODE_FILES")

OUTPUT_DIR = ROOT / "output" / "final_annotation"

IMAGE_DIR = OUTPUT_DIR / "tiles" / "images"
PREMASK_DIR = OUTPUT_DIR / "tiles" / "premasks"
MASK_DIR = OUTPUT_DIR / "tiles" / "masks"

LOG_FILE = OUTPUT_DIR / "annotation_log.csv"

MASK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

TILE_SIZE = 1024

BACKGROUND = 0
BLOB = 1
STREAK = 2

CANVAS_SIZE = 820
MIN_ZOOM = 0.50
MAX_ZOOM = 2.50


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Digantara Pixel Annotation",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# TILE DISCOVERY
# ============================================================

image_files = sorted(IMAGE_DIR.glob("*.npy"))

if not image_files:
    st.error(
        f"No tiles found in:\n{IMAGE_DIR}\n\n"
        "Run the tiling/candidate-generation pipeline first."
    )
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "tile_index": 0,
    "mask": None,
    "tile_name": None,
    "dirty": False,
    "display_mode": "Enhanced",
    "show_candidates": False,
    "show_mask": True,
    "zoom": 1.0,
    "pan_x": 512,
    "pan_y": 512,
    "history": [],
    "redo": [],
    "canvas_key": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def get_paths(index):
    image_path = image_files[index]
    tile_name = image_path.stem

    premask_path = PREMASK_DIR / f"{tile_name}.npy"
    final_mask_path = MASK_DIR / f"{tile_name}.npy"

    return image_path, premask_path, final_mask_path


def load_tile(index):
    image_path, premask_path, final_mask_path = get_paths(index)

    image = np.load(image_path).astype(np.float32)

    if image.shape != (TILE_SIZE, TILE_SIZE):
        raise ValueError(
            f"{image_path.name}: expected "
            f"{TILE_SIZE}x{TILE_SIZE}, got {image.shape}"
        )

    # Existing final annotation is loaded.
    # New tiles start with an EMPTY annotation mask.
    if final_mask_path.exists():
        mask = np.load(final_mask_path).astype(np.uint8)
        loaded_from = "existing final annotation"
    else:
        mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        loaded_from = "new empty annotation"

    mask = np.where(
        np.isin(mask, [BACKGROUND, BLOB, STREAK]),
        mask,
        BACKGROUND,
    ).astype(np.uint8)

    labels = None

    # Candidate pre-mask is ONLY used to draw optional visual boxes.
    if premask_path.exists():
        candidate = np.load(premask_path).astype(np.uint8)
        binary = (candidate > 0).astype(np.uint8)

        _, labels, _, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

    return {
        "name": image_path.stem,
        "image": image,
        "mask": mask,
        "labels": labels,
        "final_path": final_mask_path,
        "premask_path": premask_path,
        "loaded_from": loaded_from,
    }


def ensure_current_tile():
    tile_name = image_files[st.session_state.tile_index].stem

    if (
        st.session_state.mask is None
        or st.session_state.tile_name != tile_name
    ):
        tile = load_tile(st.session_state.tile_index)

        st.session_state.mask = tile["mask"].copy()
        st.session_state.tile_name = tile["name"]
        st.session_state.loaded_from = tile["loaded_from"]
        st.session_state.dirty = False
        st.session_state.history = []
        st.session_state.redo = []
        st.session_state.canvas_key += 1

    return load_tile(st.session_state.tile_index)


def normalize_image(image, low_pct=1.0, high_pct=99.8):
    low, high = np.percentile(image, [low_pct, high_pct])

    if high <= low:
        high = low + 1.0

    normalized = np.clip(
        (image - low) / (high - low),
        0,
        1,
    )

    return (normalized * 255).astype(np.uint8)


def background_subtracted(image):
    background = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=15,
    )
    return image - background


def make_base_image(tile):
    image = tile["image"]

    if st.session_state.display_mode == "Raw":
        gray = normalize_image(image, 0.5, 99.95)

    elif st.session_state.display_mode == "Enhanced":
        gray = normalize_image(image, 1.0, 99.7)

    else:
        residual = background_subtracted(image)
        gray = normalize_image(residual, 1.0, 99.7)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def make_full_display(tile):
    display = make_base_image(tile)

    # Optional candidate boxes.
    if (
        st.session_state.show_candidates
        and tile["labels"] is not None
    ):
        number, _, stats, _ = cv2.connectedComponentsWithStats(
            (tile["labels"] > 0).astype(np.uint8),
            connectivity=8,
        )

        for component_id in range(1, number):
            x, y, w, h, area = stats[component_id]

            if area < 1:
                continue

            cv2.rectangle(
                display,
                (x, y),
                (x + w - 1, y + h - 1),
                (255, 220, 0),
                1,
            )

    # Ground-truth mask overlay.
    if st.session_state.show_mask:
        mask = st.session_state.mask

        blob_pixels = mask == BLOB
        streak_pixels = mask == STREAK

        if np.any(blob_pixels):
            overlay = np.zeros_like(display)
            overlay[blob_pixels] = [0, 220, 255]
            display = cv2.addWeighted(
                display,
                0.65,
                overlay,
                0.35,
                0,
            )

        if np.any(streak_pixels):
            overlay = np.zeros_like(display)
            overlay[streak_pixels] = [255, 60, 60]
            display = cv2.addWeighted(
                display,
                0.65,
                overlay,
                0.35,
                0,
            )

    return display


def get_view(tile):
    """
    Create a zoomed/panned 820x820 view from the 1024x1024 tile.
    """
    full = make_full_display(tile)

    zoom = float(st.session_state.zoom)

    # At zoom=1, show the full tile.
    crop_size = max(32, int(round(TILE_SIZE / zoom)))
    crop_size = min(crop_size, TILE_SIZE)

    cx = int(np.clip(st.session_state.pan_x, 0, TILE_SIZE - 1))
    cy = int(np.clip(st.session_state.pan_y, 0, TILE_SIZE - 1))

    x0 = cx - crop_size // 2
    y0 = cy - crop_size // 2

    x0 = max(0, min(x0, TILE_SIZE - crop_size))
    y0 = max(0, min(y0, TILE_SIZE - crop_size))

    crop = full[y0:y0 + crop_size, x0:x0 + crop_size]

    view = cv2.resize(
        crop,
        (CANVAS_SIZE, CANVAS_SIZE),
        interpolation=cv2.INTER_NEAREST if zoom > 1 else cv2.INTER_AREA,
    )

    return view, x0, y0, crop_size


def snapshot():
    return st.session_state.mask.copy()


def push_history():
    st.session_state.history.append(snapshot())

    # Keep memory bounded.
    if len(st.session_state.history) > 30:
        st.session_state.history.pop(0)

    st.session_state.redo = []


def undo():
    if not st.session_state.history:
        st.warning("Nothing to undo.")
        return

    st.session_state.redo.append(snapshot())
    st.session_state.mask = st.session_state.history.pop()
    st.session_state.dirty = True
    st.session_state.canvas_key += 1


def redo():
    if not st.session_state.redo:
        st.warning("Nothing to redo.")
        return

    st.session_state.history.append(snapshot())
    st.session_state.mask = st.session_state.redo.pop()
    st.session_state.dirty = True
    st.session_state.canvas_key += 1


def reset_background():
    push_history()
    st.session_state.mask = np.zeros(
        (TILE_SIZE, TILE_SIZE),
        dtype=np.uint8,
    )
    st.session_state.dirty = True
    st.session_state.canvas_key += 1


def save_current_tile(tile):
    np.save(
        tile["final_path"],
        st.session_state.mask.astype(np.uint8),
    )

    record = {
        "tile_id": tile["name"],
        "mask_path": str(tile["final_path"]),
        "blob_pixels": int(np.sum(st.session_state.mask == BLOB)),
        "streak_pixels": int(np.sum(st.session_state.mask == STREAK)),
        "background_pixels": int(
            np.sum(st.session_state.mask == BACKGROUND)
        ),
        "status": "reviewed",
    }

    rows = []

    if LOG_FILE.exists():
        with open(
            LOG_FILE,
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

    rows = [
        r for r in rows
        if r.get("tile_id") != tile["name"]
    ]

    rows.append(record)

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_id",
                "mask_path",
                "blob_pixels",
                "streak_pixels",
                "background_pixels",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    st.session_state.dirty = False


def go_to_tile(new_index):
    new_index = int(
        np.clip(
            new_index,
            0,
            len(image_files) - 1,
        )
    )

    st.session_state.tile_index = new_index
    st.session_state.mask = None
    st.session_state.tile_name = None
    st.session_state.dirty = False
    st.session_state.history = []
    st.session_state.redo = []
    st.session_state.canvas_key += 1


def convert_canvas_to_mask(canvas_result, x0, y0, crop_size, mode, brush_size):
    """
    Convert the 820x820 drawable canvas into a mask for the current
    1024x1024 image crop.

    IMPORTANT:
    - canvas_result.image_data is always CANVAS_SIZE x CANVAS_SIZE.
    - crop_size is the size of the source-image crop and may be 1024,
      512, etc. depending on zoom.
    - Therefore the annotation must be detected at canvas resolution
      FIRST and resized to crop_size BEFORE indexing crop_mask.

    Returns:
        A full-tile uint8 array:
          0 = no drawing / background
          1 = blob
          2 = streak
        For eraser mode, pixels are encoded as 255 so the caller can
        distinguish "erase" from "nothing drawn".
    """
    if canvas_result is None or canvas_result.image_data is None:
        return None

    rgba = np.asarray(canvas_result.image_data)

    if rgba.ndim != 3 or rgba.shape[2] < 3:
        return None

    rgb = rgba[:, :, :3].astype(np.uint8)

    # Canvas is 820x820. Detect annotation strokes at canvas resolution.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    saturated = saturation > 110

    # Green annotation -> Blob
    blob_pixels = (
        saturated
        & (hue >= 35)
        & (hue <= 85)
        & (value >= 100)
    )

    # Magenta annotation -> Streak
    streak_pixels = (
        saturated
        & (hue >= 135)
        & (hue <= 175)
        & (value >= 100)
    )

    # Blue annotation -> Eraser
    eraser_pixels = (
        saturated
        & (hue >= 100)
        & (hue <= 135)
        & (value >= 80)
    )

    if (
        not np.any(blob_pixels)
        and not np.any(streak_pixels)
        and not np.any(eraser_pixels)
    ):
        return None

    # Work at CANVAS_SIZE first.
    canvas_mask = np.zeros(
        (CANVAS_SIZE, CANVAS_SIZE),
        dtype=np.uint8,
    )

    if mode == "Blob / Star":
        canvas_mask[blob_pixels] = BLOB

    elif mode == "Streak / Space Object":
        canvas_mask[streak_pixels] = STREAK

    else:
        canvas_mask[eraser_pixels] = 255

    # Now convert 820x820 canvas coordinates to the actual source
    # crop coordinates. This fixes the 820-vs-1024 boolean-index error.
    crop_mask = cv2.resize(
        canvas_mask,
        (crop_size, crop_size),
        interpolation=cv2.INTER_NEAREST,
    )

    # Put the crop back into the full 1024x1024 tile.
    full_mask = np.zeros_like(
        st.session_state.mask,
        dtype=np.uint8,
    )

    y1 = min(TILE_SIZE, y0 + crop_size)
    x1 = min(TILE_SIZE, x0 + crop_size)

    full_mask[
        y0:y1,
        x0:x1,
    ] = crop_mask[
        :y1 - y0,
        :x1 - x0,
    ]

    return full_mask


# ============================================================
# LOAD CURRENT TILE
# ============================================================

tile = ensure_current_tile()

tile_number = st.session_state.tile_index + 1
total_tiles = len(image_files)


# ============================================================
# HEADER
# ============================================================

st.title("🛰️ Digantara — Manual Pixel Annotation")

st.caption(
    "Fully manual pixel-level annotation for stars/blobs and "
    "space-object streaks in 1024 × 1024 tiles."
)

st.info(
    "Paint directly over any object you can see. "
    "Candidate boxes are optional visual hints only and do not "
    "limit where you can annotate."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🎨 Annotation Controls")

    mode = st.radio(
        "Annotation class",
        [
            "Blob / Star",
            "Streak / Space Object",
            "Eraser / Background",
        ],
        index=0,
    )

    if mode == "Blob / Star":
        stroke_color = "#00FF00"
        cursor_help = "Paint point-like stars/blobs."

    elif mode == "Streak / Space Object":
        stroke_color = "#FF00FF"
        cursor_help = "Drag along the full visible streak."

    else:
        stroke_color = "#0000FF"
        cursor_help = "Paint over an incorrect annotation to erase it."

    brush_size = st.slider(
        "Brush size (pixels)",
        min_value=1,
        max_value=40,
        value=5,
    )

    st.caption(cursor_help)

    st.divider()

    st.subheader("🔍 View")

    st.session_state.display_mode = st.radio(
        "Image view",
        [
            "Enhanced",
            "Raw",
            "Background-subtracted",
        ],
        index=[
            "Enhanced",
            "Raw",
            "Background-subtracted",
        ].index(st.session_state.display_mode),
    )

    st.session_state.zoom = st.slider(
        "Zoom",
        min_value=MIN_ZOOM,
        max_value=MAX_ZOOM,
        value=float(st.session_state.zoom),
        step=0.25,
    )

    if st.session_state.zoom <= 1.0:
        st.session_state.pan_x = 512
        st.session_state.pan_y = 512

    else:
        st.session_state.pan_x = st.slider(
            "Pan X",
            0,
            TILE_SIZE - 1,
            int(st.session_state.pan_x),
        )

        st.session_state.pan_y = st.slider(
            "Pan Y",
            0,
            TILE_SIZE - 1,
            int(st.session_state.pan_y),
        )

    st.session_state.show_candidates = st.checkbox(
        "Show candidate boxes",
        value=st.session_state.show_candidates,
    )

    st.session_state.show_mask = st.checkbox(
        "Show annotation overlay",
        value=st.session_state.show_mask,
    )

    st.divider()

    st.subheader("↶ Editing")

    undo_col, redo_col = st.columns(2)

    with undo_col:
        if st.button(
            "↶ Undo",
            use_container_width=True,
            disabled=not st.session_state.history,
        ):
            undo()
            st.rerun()

    with redo_col:
        if st.button(
            "↷ Redo",
            use_container_width=True,
            disabled=not st.session_state.redo,
        ):
            redo()
            st.rerun()

    if st.button(
        "🗑 Clear Annotation",
        use_container_width=True,
    ):
        reset_background()
        st.rerun()

    st.divider()

    st.subheader("🗂 Tile Navigation")

    selected_tile = st.number_input(
        "Tile index",
        min_value=1,
        max_value=total_tiles,
        value=tile_number,
        step=1,
    )

    if selected_tile != tile_number:
        if st.session_state.dirty:
            st.warning("Save the current tile before changing tiles.")
        else:
            go_to_tile(selected_tile - 1)
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "⬅ Previous",
            use_container_width=True,
            disabled=tile_number == 1 or st.session_state.dirty,
        ):
            go_to_tile(tile_number - 2)
            st.rerun()

    with col2:
        if st.button(
            "Next ➡",
            use_container_width=True,
            disabled=tile_number == total_tiles or st.session_state.dirty,
        ):
            go_to_tile(tile_number)
            st.rerun()


# ============================================================
# TILE INFO
# ============================================================

st.subheader(f"Tile {tile_number} / {total_tiles}")

st.code(tile["name"])

if st.session_state.dirty:
    st.warning("⚠️ Unsaved changes")
else:
    st.success(f"Loaded: {st.session_state.loaded_from}")


# ============================================================
# STATISTICS
# ============================================================

mask = st.session_state.mask

blob_count = int(np.sum(mask == BLOB))
streak_count = int(np.sum(mask == STREAK))
background_count = int(np.sum(mask == BACKGROUND))

c1, c2, c3, c4 = st.columns(4)

c1.metric("Background", f"{background_count:,}")
c2.metric("Blob / Star", f"{blob_count:,}")
c3.metric("Streak", f"{streak_count:,}")
c4.metric("Tile size", "1024 × 1024")


# ============================================================
# INSTRUCTIONS
# ============================================================

st.markdown(
    f"""
**Current mode:** `{mode}` &nbsp;&nbsp; **Brush:** `{brush_size}px` &nbsp;&nbsp;
**Zoom:** `{st.session_state.zoom:.2f}×`

- 🟩 **Blob / Star:** click or drag over a point source.
- 🟪 **Streak / Space Object:** hold the mouse button and trace the complete streak.
- 🟦 **Eraser:** paint over an incorrect annotation.
- 🔍 **Zoom:** zoom in to inspect faint objects.
- 🧭 **Pan:** when zoomed in, use Pan X / Pan Y.
- ↶ **Undo / Redo:** correct accidental strokes.
- Yellow candidate boxes are only suggestions. **Inspect the entire tile.**
"""
)


# ============================================================
# DRAWING CANVAS
# ============================================================

view, crop_x, crop_y, crop_size = get_view(tile)

view_pil = Image.fromarray(view)

st.subheader("🖌️ Annotation Canvas")

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=max(1, int(round(brush_size * st.session_state.zoom))),
    stroke_color=stroke_color,
    background_image=view_pil,
    update_streamlit=True,
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
    drawing_mode="freedraw",
    return_image_data=True,
    key=(
        f"canvas_"
        f"{re.sub(r'[^A-Za-z0-9-]+', '-', tile['name']).strip('-')}"
        f"-{st.session_state.canvas_key}"
    ),
)


# ============================================================
# APPLY DRAWING
# ============================================================

if canvas_result is not None and canvas_result.image_data is not None:
    current_signature = (
        canvas_result.image_data.shape,
        int(np.sum(canvas_result.image_data)),
    )

    if st.session_state.get("last_canvas_signature") != current_signature:
        st.session_state.last_canvas_signature = current_signature

        painted = convert_canvas_to_mask(
            canvas_result,
            crop_x,
            crop_y,
            crop_size,
            mode,
            brush_size,
        )

        if painted is not None and np.any(painted):
            push_history()

            if mode == "Eraser / Background":
                # Eraser pixels are encoded as 255 by
                # convert_canvas_to_mask().
                erase_full = painted == 255
                st.session_state.mask[erase_full] = BACKGROUND

            else:
                # Blob = 1, Streak = 2.
                annotation_pixels = painted > 0
                st.session_state.mask[annotation_pixels] = painted[
                    annotation_pixels
                ]

            st.session_state.dirty = True
            st.session_state.canvas_key += 1
            st.rerun()


# ============================================================
# ACTION BUTTONS
# ============================================================

st.divider()

left, middle, right = st.columns(3)

with left:
    if st.button(
        "💾 SAVE FINAL MASK",
        type="primary",
        use_container_width=True,
    ):
        save_current_tile(tile)
        st.success(f"Saved mask for {tile['name']}")
        st.session_state.canvas_key += 1
        st.rerun()

with middle:
    if st.button(
        "↩ Reset to Background",
        use_container_width=True,
    ):
        reset_background()
        st.rerun()

with right:
    if st.button(
        "🔄 Refresh View",
        use_container_width=True,
    ):
        st.session_state.canvas_key += 1
        st.rerun()


# ============================================================
# LEGEND
# ============================================================

st.divider()

st.markdown(
    """
### Mask legend

| Value | Class | Meaning |
|---:|---|---|
| 0 | Background | Noise / empty sky / rejected pixels |
| 1 | Blob / Star | Point-like source |
| 2 | Streak / Space Object | Elongated linear source |

**Important:** the final `.npy` file contains only the manually created
pixel-level mask. The original FITS-derived tile is never modified.
"""
)


# ============================================================
# PROGRESS
# ============================================================

completed = 0

if LOG_FILE.exists():
    try:
        log_df = pd.read_csv(LOG_FILE)
        if "status" in log_df.columns:
            completed = int(
                (log_df["status"] == "reviewed").sum()
            )
    except Exception:
        completed = 0

st.progress(
    min(completed / total_tiles, 1.0),
    text=f"Annotation progress: {completed}/{total_tiles} tiles reviewed",
)
