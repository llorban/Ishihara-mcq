#!/usr/bin/env python3
"""
generate_mcq_images_fontcontrols.py

Generate 700x300 "Ishihara-style" MCQ plates (plain PNGs), with explicit
font settings you can play with (auto-fit or fixed sizes).

Usage examples:

# Use built-in sample questions (12 plates)
python generate_mcq_images_fontcontrols.py --count 12 --out plates_out --zip

# Use CSV file with header: question,optA,optB,optC,optD
python generate_mcq_images_fontcontrols.py --csv my_questions.csv --out plates_out --debug

# Force question font size to 40 and option font size to 22 (px)
python generate_mcq_images_fontcontrols.py --count 6 --q-font-size 40 --opt-font-size 22 --out plates_out

# Disable auto-fit (i.e., use forced sizes or fallback)
python generate_mcq_images_fontcontrols.py --no-auto-fit --q-font-size 48 --opt-font-size 20 --out plates_out

Dependencies:
  pip install pillow numpy
"""

import os, sys, argparse, textwrap, csv, json, random, zipfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi   # pip install scipy

def elastic_warp(mask_pil, alpha=24, sigma=6, seed=None):
    """
    Return a *warped* mask PIL image using random displacement fields.
    alpha: max displacement magnitude (px)
    sigma: smoothing (px)
    """
    if seed is not None:
        np.random.seed(seed)
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    shape = mask_np.shape
    dx = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    # build coordinate grid and displace
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    warped = ndi.map_coordinates(mask_np, [map_y.ravel(), map_x.ravel()], order=1, mode='reflect')
    warped = warped.reshape(shape)
    return Image.fromarray((np.clip(warped,0,1)*255).astype('uint8'), mode='L')

def random_occlusions_on_mask(mask_pil, p=0.02, size_range=(2,8), seed=None):
    """
    Randomly erase tiny clusters in the mask to simulate occlusions.
    p: fraction of mask pixels to consider for occlusion
    size_range: radius range (px) for occluding circles
    """
    if seed is not None:
        random.seed(seed)
    mask = mask_pil.copy()
    draw = ImageDraw.Draw(mask)
    w,h = mask.size
    total = int(w*h*p)
    for _ in range(total):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        r = random.randint(size_range[0], size_range[1])
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=0)
    return mask

def sample_fg_with_variation(mask_np, dot_positions, fg_color=(200,40,40),
                             fg_size=8, color_jitter=14, seed=None, drop_chance=0.02):
    """
    Return list of (cx,cy, rr, rgba_color) for FG dots given candidate dot positions.
    - mask_np: 2D numpy array (H x W) of mask values in [0..1]
    - dot_positions: iterable of (cx,cy) pixel coordinates to consider
    - fg_color: base RGB tuple
    - fg_size: nominal radius in px
    - color_jitter: +/- per-channel jitter
    - seed: optional seed for deterministic randomness
    - drop_chance: probability to randomly skip a foreground dot (simulates occlusion/noise)
    """
    if seed is not None:
        random.seed(seed)
    draws = []
    h, w = mask_np.shape
    for (cx, cy) in dot_positions:
        # skip out-of-bounds gracefully
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        v = mask_np[int(cy), int(cx)]
        if v <= MASK_THRESHOLD:
            continue
        # randomly drop a small fraction of FG dots to introduce gaps
        if drop_chance and (random.random() < float(drop_chance)):
            continue
        rr = fg_size + random.randint(-1, 1)  # slight size jitter
        r = max(0, min(255, fg_color[0] + random.randint(-color_jitter, color_jitter)))
        g = max(0, min(255, fg_color[1] + random.randint(-color_jitter, color_jitter)))
        b = max(0, min(255, fg_color[2] + random.randint(-color_jitter, color_jitter)))
        draws.append((cx, cy, rr, (r, g, b, 255)))
    return draws

# ---------------- Default visual/dot tuning (changeable) -----------------
WIDTH, HEIGHT = 1400, 700
DOT_PITCH = 1        # smaller => denser
JITTER = 1 
BG_DOT_SIZE = 2
FG_DOT_SIZE = 1 
BG_PASSES = 4
FG_PASSES = 4
MASK_THRESHOLD = 0.05 # mask / placement

# obfuscation
ELASTIC_ALPHA = 2  # DISPLACEMENT SCALE (PX)
ELASTIC_SIGMA = 3   # SMOOTHNESS (PX)
OCCLUDE_P = 0.002   # FRACTION OF MASK PIXELS RANDOMLY ERASED (1.5%)
OCCLUDE_SIZE_RANGE = (1, 3)
ROTATE_DEG = 2    # SMALL ROTATION +/- DEG
SHEAR_DEG = 2     # SMALL SHEAR +/- DEG

# color jitter
FG_COLOR_JITTER = 16  # +/- per channel
BG_COLOR_JITTER = 6

# enable saving/logging of per-plate random seed and obfuscation parameters
LOG_SEED_AND_PARAMS = True

JSON_MAX_FILES = 1000
JSON_MAX_FILE_BYTES = 1_000_000

# ------------------------------------------------------------------------

# ---------------- Default font settings & search dirs -------------------
FONT_SEARCH_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/Library/Fonts"),
    "C:\\Windows\\Fonts"
]
DEFAULT_Q_AREA_RATIO = 0.44   # fraction of height reserved for the question block
OPTION_PADDING = 3            # bbox padding around detected option text
Q_PADDING = 8                 # padding used when fitting question font
# ------------------------------------------------------------------------

def find_any_ttf(search_dirs=None):
    search_dirs = search_dirs or FONT_SEARCH_DIRS
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        name = f.lower()
                        # skip likely emoji/color fonts which can behave oddly
                        if "emoji" in name or "color" in name:
                            continue
                        return os.path.join(root, f)
        except PermissionError:
            continue
    return None

def text_size(draw, text, font):
    """Robust text measurement returning (w,h)."""
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text)*8, getattr(font, "size", 12))

def largest_fitting_truetype(lines, w, h, padding=8, font_path=None, max_hint=None):
    """
    Binary-search largest truetype font size that fits `lines` into w x h (approx).
    Returns (ImageFont instance, chosen_size).
    If font_path is None -> returns default PIL font (non-scalable).
    max_hint optional ceiling (e.g. int(h*0.9)).
    """
    draw_tmp = ImageDraw.Draw(Image.new("L", (w, h)))
    if font_path is None:
        f = ImageFont.load_default()
        return f, getattr(f, "size", 12)
    lo, hi = 8, max_hint if max_hint else int(h * 1.2)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            font = ImageFont.truetype(font_path, mid)
        except Exception:
            hi = mid - 1
            continue
        wmax = 0; hsum = 0
        for ln in lines:
            wln, hln = text_size(draw_tmp, ln, font)
            wmax = max(wmax, wln)
            hsum += hln + 4
        if (wmax <= w - 2*padding) and (hsum <= h - 2*padding):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    try:
        return ImageFont.truetype(font_path, best), best
    except Exception:
        f = ImageFont.load_default()
        return f, getattr(f, "size", 12)

def build_mask_and_boxes(question, options, font_path, auto_fit=True, q_forced_size=None, opt_forced_size=None, q_padding=Q_PADDING, option_padding=OPTION_PADDING):
    """
    Render the whole WIDTHxHEIGHT mask with question + options.
    Returns: PIL L mask and list of option bounding boxes [ [x0,y0,x1,y1], ... ]
    Behavior:
      * If auto_fit True: pick largest font that fits question area and options area.
      * If forced sizes provided (q_forced_size / opt_forced_size) they are used.
    """
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)

    # --- question layout ---
    q_lines = textwrap.wrap(question, width=60)
    q_area_h = int(HEIGHT * DEFAULT_Q_AREA_RATIO)
    opt_area_h = HEIGHT - q_area_h - 20

    # choose question font
    q_font = None
    q_size = None
    if q_forced_size:
        try:
            q_font = ImageFont.truetype(font_path, q_forced_size) if font_path else ImageFont.load_default()
            q_size = q_forced_size
        except Exception:
            q_font = ImageFont.load_default()
            q_size = getattr(q_font, "size", 12)
    else:
        if auto_fit:
            q_font, q_size = largest_fitting_truetype(q_lines, WIDTH - 24, q_area_h, padding=q_padding, font_path=font_path, max_hint=int(q_area_h * 0.9))
        else:
            # fallback sized font
            q_font = ImageFont.truetype(font_path, max(18, int(q_area_h * 0.12))) if font_path else ImageFont.load_default()
            q_size = getattr(q_font, "size", 12)

    # draw question centered at top block
    y = 10
    for ln in q_lines:
        w, h = text_size(draw, ln, q_font)
        draw.text(((WIDTH - w)//2, y), ln, fill=255, font=q_font)
        y += h + 4

    # --- options layout (REPLACEMENT: full-width rows with pixel-wrap) ---
    # We'll compute a single opt font that fits all options into equal-height rows,
    # then draw each option filling the full horizontal span and return full-width boxes.

    def wrap_text_by_pixel(draw_obj, text, font_obj, max_width):
        """Wrap a text string to lines whose pixel width <= max_width using draw_obj.textbbox."""
        words = text.split()
        if not words:
            return [""]
        lines = []
        cur = words[0]
        for wword in words[1:]:
            test = cur + " " + wword
            bbox = draw_obj.textbbox((0,0), test, font=font_obj)
            if bbox[2] - bbox[0] <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = wword
        lines.append(cur)
        return lines

    def find_single_options_font(draw_obj, options_list, font_path, box_width, per_row_h, padding=8, min_size=8, max_size_hint=72):
        """
        Binary-search / scan for a single font size that fits EVERY option (when pixel-wrapped)
        into the per_row_h band and box_width.
        Returns (font_obj, chosen_size, wrapped_options_lines) or (None, None, None) if no fit.
        """
        inner_w = max(1, box_width - 2*padding)
        inner_h = max(1, per_row_h - 2*padding)
        if inner_w <= 0 or inner_h <= 0:
            return None, None, None

        # scan from large -> small (keep current behavior of preferring larger readable text)
        for size_test in range(max_size_hint, min_size-1, -1):
            try:
                ftest = ImageFont.truetype(font_path, size_test) if font_path else ImageFont.load_default()
            except Exception:
                ftest = ImageFont.load_default()
            ok = True
            wrapped_all = []
            for opt in options_list:
                lines_opt = wrap_text_by_pixel(draw_obj, opt, ftest, inner_w)
                # measure height required
                total_h = 0
                too_wide = False
                for ln in lines_opt:
                    b = draw_obj.textbbox((0,0), ln, font=ftest)
                    lw = b[2] - b[0]
                    lh = b[3] - b[1]
                    total_h += lh + 2
                    if lw > inner_w:
                        too_wide = True
                if too_wide or total_h > inner_h:
                    ok = False
                    break
                wrapped_all.append(lines_opt)
            if ok:
                return ftest, size_test, wrapped_all
        return None, None, None

    # compute layout geometry
    per_row_h = int(opt_area_h / max(1, len(options)))  # even split vertically
    left_margin = max(8, q_padding) if 'q_padding' in locals() else 14
    right_margin = left_margin
    opt_x0 = left_margin
    opt_x1 = WIDTH - right_margin
    opt_box_w = opt_x1 - opt_x0

    # pick a font that fits all options (pixel-aware)
    opt_font, opt_chosen_size, wrapped_options = find_single_options_font(draw, options, font_path,
                                                                          opt_box_w, per_row_h,
                                                                          padding=option_padding if 'option_padding' in locals() else 8,
                                                                          min_size=8, max_size_hint=60)

    # fallback to 2-column if single-column didn't fit
    boxes = []
    if opt_font is None:
        # two-column layout attempt
        cols = 2
        rows_per_col = (len(options) + cols - 1) // cols
        col_w = (WIDTH - left_margin - right_margin - 8) // cols
        per_row_h_col = int(options_band_h / rows_per_col) if 'options_band_h' in locals() else per_row_h
        col_font, col_size, col_wrapped = find_single_options_font(draw, options, font_path, col_w, per_row_h_col,
                                                                  padding=6, min_size=7, max_size_hint=48)
        if col_font is None:
            # final fallback: use the old behaviour (tiny font) - compute simple wrapped lines with default font
            col_font = ImageFont.truetype(font_path, 12) if font_path else ImageFont.load_default()
            col_wrapped = [textwrap.wrap(opt, width=40) for opt in options]
        # draw two-column
        draw_x_gutter = 8
        for idx, opt in enumerate(options):
            col_idx = idx // rows_per_col
            row_idx = idx % rows_per_col
            x0 = left_margin + col_idx * (col_w + draw_x_gutter)
            y0 = (y + 8) + row_idx * per_row_h_col
            # wrap for this col with the chosen col_font
            lines = wrap_text_by_pixel(draw, opt, col_font, col_w - 2*6)
            yy = y0 + 6
            for ln in lines:
                draw.text((x0 + 6, yy), ln, fill=255, font=col_font)
                b = draw.textbbox((x0 + 6, yy), ln, font=col_font)
                yy += (b[3] - b[1]) + 2
            boxes.append([x0, y0, x0 + col_w, y0 + per_row_h_col])
    else:
        # draw single-column full-width rows
        for i, lines in enumerate(wrapped_options):
            row_y0 = (y + 8) + i * per_row_h
            yy = row_y0 + 8
            # left-align the text inside the row (but the hotspot will be full-row)
            line_x = opt_x0 + 8
            for ln in lines:
                draw.text((line_x, yy), ln, fill=255, font=opt_font)
                b = draw.textbbox((line_x, yy), ln, font=opt_font)
                yy += (b[3] - b[1]) + 2
            boxes.append([opt_x0, row_y0, opt_x1, row_y0 + per_row_h])

    # end replacement options layout

    # final smoothing
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.0))

    # debug info about chosen fonts/sizes
    debug_info = {
        "q_font_size": q_size,
        "opt_font_size": opt_chosen_size,
        "q_font_used": font_path or "PIL-default",
        "q_area_h": q_area_h,
        "per_row_h": per_row_h
    }
    return mask, boxes, debug_info

def generate_plate_image(question, options, outpath, font_path, auto_fit=True, q_size_override=None, opt_size_override=None, debug=False, seed_override=None):
    # --- existing: mask, boxes = build_mask_and_option_boxes(...)
    mask, boxes, debug_info = build_mask_and_boxes(question, options, font_path, auto_fit=auto_fit, q_forced_size=q_size_override, opt_forced_size=opt_size_override)
    # create a reproducible seed and log it for debugging
    seed = random.randint(0, 2**31 - 1)
    if LOG_SEED_AND_PARAMS:
        # append or save a small JSON later per-plate (we'll add after saving)
        debug_info['seed'] = seed
        meta = {
          "question": question,
          "options": options,
          "boxes": boxes,
          "font_info": debug_info,
          "seed": seed,
          "obfuscation": {
            "ELASTIC_ALPHA": ELASTIC_ALPHA,
            "ELASTIC_SIGMA": ELASTIC_SIGMA,
            "OCCLUDE_P": OCCLUDE_P,
            "FG_PASSES": FG_PASSES,
            "FG_COLOR_JITTER": FG_COLOR_JITTER,
            "ROTATE_DEG": ROTATE_DEG,
            "SHEAR_DEG": SHEAR_DEG
          }
        }
        # save as before

    # 1) elastic warp (non-linear displacement) to mask
    mask = elastic_warp(mask, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA, seed=seed)
    # 2) random tiny occlusions (erase small circles)
    mask = random_occlusions_on_mask(mask, p=OCCLUDE_P, size_range=OCCLUDE_SIZE_RANGE, seed=seed)
    # then sample mask into float array used for dot selection
    mask_np = np.array(mask).astype(float) / 255.0

    # optionally save debug mask if requested (keep your debug behavior)
    if debug:
        mask.save(outpath.replace(".png", "_mask.png"))

    # background
    bg = Image.new("RGB", (WIDTH, HEIGHT), (240,240,240))
    for pass_i in range(BG_PASSES):
        draw_bg = ImageDraw.Draw(bg)
        offset_x = (pass_i * (DOT_PITCH // 3)) % DOT_PITCH
        offset_y = (pass_i * (DOT_PITCH // 5)) % DOT_PITCH
        cols = int(WIDTH / DOT_PITCH) + 2
        rows = int(HEIGHT / DOT_PITCH) + 2
        for r in range(rows):
            for c in range(cols):
                cx = int(c * DOT_PITCH + DOT_PITCH/2 + offset_x - (cols*DOT_PITCH - WIDTH)/2)
                cy = int(r * DOT_PITCH + DOT_PITCH/2 + offset_y - (rows*DOT_PITCH - HEIGHT)/2)
                jx = cx + random.randint(-JITTER, JITTER)
                jy = cy + random.randint(-JITTER, JITTER)
                if not (0 <= jx < WIDTH and 0 <= jy < HEIGHT): continue
                rr = BG_DOT_SIZE
                col = (180 + ((r+c+pass_i) % 3)*10, 180 + ((r+c+pass_i)%3)*6, 60 + ((r+c+pass_i)%3)*20)
                draw_bg.ellipse([jx-rr/2, jy-rr/2, jx+rr/2, jy+rr/2], fill=col, outline=None)

    # --------- foreground: multi-pass sampling with color/size jitter & occasional drops ----------
    fg = Image.new("RGBA", (WIDTH, HEIGHT), (0,0,0,0))
    draw_fg = ImageDraw.Draw(fg)
    cols = int(WIDTH / DOT_PITCH) + 2
    rows = int(HEIGHT / DOT_PITCH) + 2

    # build base candidate dot positions once
    base_positions = []
    for r in range(rows):
        for c in range(cols):
            cx = int(c * DOT_PITCH + DOT_PITCH/2 - (cols*DOT_PITCH - WIDTH)/2)
            cy = int(r * DOT_PITCH + DOT_PITCH/2 - (rows*DOT_PITCH - HEIGHT)/2)
            base_positions.append((cx, cy))

    # multiple passes with slight offsets to reduce aliasing + increase fill
    for pass_i in range(FG_PASSES):
        offset_x = (pass_i * (DOT_PITCH // 3)) % DOT_PITCH
        offset_y = (pass_i * (DOT_PITCH // 5)) % DOT_PITCH
        # build pass positions with offset (so slightly different samples each pass)
        pass_positions = [(x + offset_x, y + offset_y) for (x, y) in base_positions]
        # sample draws for this pass
        draw_cmds = sample_fg_with_variation(mask_np, pass_positions,
                                            fg_color=(200, 40, 40),
                                            fg_size=FG_DOT_SIZE,
                                            color_jitter=FG_COLOR_JITTER,
                                            seed=seed + pass_i, drop_chance=0.02)
        # render draws
        for (cx, cy, rr, color) in draw_cmds:
            draw_fg.ellipse([cx-rr/2, cy-rr/2, cx+rr/2, cy+rr/2], fill=color, outline=None)
    # save fg debug if requested
    if debug:
        fg.convert("RGB").save(outpath.replace(".png", "_fg.png"))
    # ------------------------------------------------------------------------

    composite = Image.alpha_composite(bg.convert("RGBA"), fg).convert("RGB")
    # small global transform to further break exact text geometry
    angle = random.uniform(-ROTATE_DEG, ROTATE_DEG)
    shear = random.uniform(-SHEAR_DEG, SHEAR_DEG)
    composite = composite.rotate(angle, resample=Image.BICUBIC, expand=False)
    # shear via AFFINE matrix (approx): (a, b, c, d, e, f) maps
    composite = composite.transform((WIDTH, HEIGHT), Image.AFFINE,
                                    (1, np.tan(np.radians(shear)), 0, 0, 1, 0),
                                    resample=Image.BICUBIC)
    composite.save(outpath)
    if debug:
        mask.save(outpath.replace(".png", "_mask.png"))
        bg.save(outpath.replace(".png", "_bg.png"))
        fg.convert("RGB").save(outpath.replace(".png", "_fg.png"))
    return boxes, debug_info

def load_questions_from_csv(csv_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r.get('question') or r.get('Question') or ""
            opts = [
                r.get('optA') or r.get('A') or r.get('opt1') or r.get('optionA') or r.get('option1') or "",
                r.get('optB') or r.get('B') or r.get('opt2') or r.get('optionB') or r.get('option2') or "",
                r.get('optC') or r.get('C') or r.get('opt3') or r.get('optionC') or r.get('option3') or "",
                r.get('optD') or r.get('D') or r.get('opt4') or r.get('optionD') or r.get('option4') or ""
            ]
            rows.append((q, opts))
    return rows

def validate_plate_obj(obj, src_name):
    """Return (question, options, meta) or raise ValueError with a helpful message."""
    if not isinstance(obj, dict):
        raise ValueError(f"{src_name}: each plate must be a JSON object")
    q = obj.get("question") or obj.get("Question")
    opts = obj.get("options") or obj.get("opts") or obj.get("choices")
    if q is None or not isinstance(q, str) or q.strip() == "":
        raise ValueError(f"{src_name}: missing or invalid 'question' (must be non-empty string)")
    if not isinstance(opts, list) or len(opts) < 2:
        raise ValueError(f"{src_name}: 'options' must be a list (2..n entries). Found: {type(opts)} / length {len(opts) if isinstance(opts, list) else 'NA'}")
    # prefer exactly 4 options for MCQ, but allow 2..6 — you may enforce exactly 4 later
    if len(opts) != 4:
        # warn but accept
        print(f"WARNING: {src_name}: options length is {len(opts)} (recommended: 4).")
    # normalize options to strings
    opts_clean = [str(x) for x in opts]
    meta = {
        "id": obj.get("id") or obj.get("name"),
        "correct_index": obj.get("correct_index"),
        "seed": obj.get("seed")
    }
    return q.strip(), opts_clean, meta

def load_questions_from_json_dir(dirpath):
    """
    Load JSON files from a directory. Returns list of (q, options, meta).
    - Accepts .json files only (ignores others)
    - Sorts files alphabetically for deterministic order
    - Validates each file
    """
    p = Path(dirpath)
    if not p.exists():
        raise SystemExit(f"JSON directory not found: {dirpath}")
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".json"])
    if not files:
        raise SystemExit(f"No .json files found in directory: {dirpath}")
    if len(files) > JSON_MAX_FILES:
        raise SystemExit(f"Too many JSON files ({len(files)}). Limit is {JSON_MAX_FILES}.")
    out = []
    for f in files:
        b = f.stat().st_size
        if b > JSON_MAX_FILE_BYTES:
            raise SystemExit(f"JSON file too large: {f.name} ({b} bytes) — limit {JSON_MAX_FILE_BYTES}")
        try:
            with f.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception as e:
            raise SystemExit(f"Failed to parse JSON {f.name}: {e}")
        # Support either object-per-file (single plate) or array (multiple plates)
        if isinstance(obj, list):
            for idx, sub in enumerate(obj):
                try:
                    q, opts, meta = validate_plate_obj(sub, f"{f.name}[{idx}]")
                    # attach source filename for traceability
                    meta["source_file"] = f.name
                    out.append((q, opts, meta))
                except ValueError as ve:
                    raise SystemExit(str(ve))
        else:
            try:
                q, opts, meta = validate_plate_obj(obj, f.name)
                meta["source_file"] = f.name
                out.append((q, opts, meta))
            except ValueError as ve:
                raise SystemExit(str(ve))
    print(f"Loaded {len(out)} plates from {dirpath}")
    return out

def load_questions_from_json_file(filepath):
    """
    Load a single JSON file that contains an array of plate objects.
    Returns list of (q, opts, meta).
    """
    p = Path(filepath)
    if not p.exists():
        raise SystemExit(f"JSON file not found: {filepath}")
    b = p.stat().st_size
    if b > JSON_MAX_FILE_BYTES:
        raise SystemExit(f"JSON file too large: {p.name} ({b} bytes).")
    try:
        with p.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception as e:
        raise SystemExit(f"Failed to parse JSON {p.name}: {e}")
    if not isinstance(obj, list):
        raise SystemExit(f"JSON file must contain a top-level array of plate objects.")
    out = []
    for idx, sub in enumerate(obj):
        q, opts, meta = validate_plate_obj(sub, f"{p.name}[{idx}]")
        meta["source_file"] = p.name
        out.append((q, opts, meta))
    print(f"Loaded {len(out)} plates from {filepath}")
    return out

def main():
    parser = argparse.ArgumentParser(description="Generate MCQ plates with font controls")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out", default="plates_out")
    parser.add_argument("--csv", help="CSV file with header (question,optA,optB,optC,optD). If absent, sample questions are used.")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Also save mask/bg/fg for each plate")
    parser.add_argument("--font-path", help="Explicit TTF/OTF path to use for all rendering (overrides auto-detect)")
    parser.add_argument("--auto-fit-fonts", dest="auto_fit", action="store_true", help="Auto-fit fonts to available space (default)")
    parser.add_argument("--no-auto-fit", dest="auto_fit", action="store_false", help="Disable auto-fit; uses forced sizes or reasonable defaults")
    parser.add_argument("--q-font-size", type=int, help="Force question font absolute size (px); overrides auto-fit if provided")
    parser.add_argument("--opt-font-size", type=int, help="Force options font absolute size (px); overrides auto-fit if provided")
    parser.add_argument("--json-dir", help="Directory containing per-plate JSON files")
    parser.add_argument("--json-file", help="Single JSON file containing array of plate objects")
    parser.set_defaults(auto_fit=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # find font if not provided
    font_path = args.font_path or find_any_ttf()
    if font_path:
        print("Using TTF font:", font_path)
    else:
        print("WARNING: No TTF/OTF found in search dirs. PIL default font will be used (likely small).", file=sys.stderr)
        font_path = None

    questions_meta = []
    if args.json_dir:
        questions_meta = load_questions_from_json_dir(args.json_dir)
    elif args.json_file:
        questions_meta = load_questions_from_json_file(args.json_file)
    elif args.csv:
        rows = load_questions_from_csv(args.csv)
        # convert to same (q,opts,meta) format (meta contains no seed/id)
        questions_meta = [(q, opts, {"source_file": args.csv}) for (q, opts) in rows]
    else:
        # fallback to built-in sample list (wrap items into meta)
        sample = [ ... ]  # your existing sample
        questions_meta = [(q, opts, {"source_file": "builtin"}) for (q, opts) in sample[:args.count]]

    written = []
    for i, (qtext, opts, meta) in enumerate(questions_meta, start=1):
        # choose a filename base
        fname_base = meta.get("id") or f"plate_{i:02d}"
        outname = os.path.join(args.out, fname_base + ".png")
        # if meta contains 'seed', use it to make the plate reproducible
        seed_override = meta.get("seed")
        boxes, info = generate_plate_image(qtext, opts, outname, font_path=font_path,
                                           auto_fit=args.auto_fit,
                                           q_size_override=args.q_font_size,
                                           opt_size_override=args.opt_font_size,
                                           debug=args.debug,
                                           seed_override=seed_override)  # you may need to thread seed into generate_plate_image
        # save meta JSON next to PNG
        meta_out = {"question": qtext, "options": opts, "boxes": boxes, "font_info": info}
        if meta.get("correct_index") is not None:
            meta_out["correct_index"] = meta["correct_index"]
        if meta.get("seed") is not None:
            meta_out["seed"] = meta["seed"]
        with open(outname.replace(".png", ".json"), "w", encoding="utf-8") as fh:
             json.dump(meta_out, fh, indent=2, ensure_ascii=False)

    if args.zip:
        zipname = os.path.join(args.out, "plates.zip")
        with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pth in written:
                zf.write(pth, arcname=os.path.basename(pth))
        print("Wrote ZIP:", zipname)

    print("Done. Open the output folder to upload PNGs into Moodle.")
    print("Validation tips: view masks (if --debug) at 100% zoom to confirm text sizes; adjust --q-font-size / --opt-font-size or toggle --no-auto-fit if needed.")

if __name__ == "__main__":
    main()
