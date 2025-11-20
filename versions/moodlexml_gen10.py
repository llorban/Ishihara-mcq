#!/usr/bin/env python3
"""
plates_to_moodle_xml_square.py

Generate a Moodle XML package with square target-boxes that fully contain the
correct-option region from per-plate JSON files.

Usage:
  python3 plates_to_moodle_xml_square.py --json-dir ./json_out --img-dir ./plates_out --out moodle_square.zip --pad 4

Outputs:
 - moodle_square.zip (questions.xml + PNGs + plates_moodle_coords_square.csv)
 - questions.xml (for inspection)
 - plates_moodle_coords_square.csv (image, json, qname, correct_index, left,top,size,right,bottom)
"""
import os, sys, argparse, json, base64, glob, csv, html
from pathlib import Path
from collections import Counter

# configure allowed extensions and preference order (deterministic)
ALLOWED_EXTS = ("png", "PNG", "jpg", "jpeg", "JPG", "JPEG")
# preferred order when multiple matches exist (you can change)
PREFERRED_ORDER = ("png","PNG","jpg","JPG","jpeg","JPEG")


def sanitize_name(s):
    import re
    tokens = re.findall(r"[A-Za-z0-9']+", s)
    if not tokens:
        return "plate"
    name = "_".join(tokens[:7])
    return name[:60]

def embed_file_base64(path):
    with open(path, "rb") as fh:
        data = fh.read()
    return base64.b64encode(data).decode("ascii")

def find_image_for_json_strict(jf_path: Path, img_dir: Path):
    """
    Strict matching by identical basename, but deduplicate candidates that
    resolve to the same file on case-insensitive filesystems.
    Returns (chosen_real_path_or_None, list_of_matched_paths_seen_by_caller).
    The returned chosen path is the filesystem-resolved Path (so .name will be
    the actual on-disk filename).
    """
    stem = jf_path.stem
    # Map canonical/resolved path -> an example Path that was used to find it
    resolved_map = {}
    # preserve order of discovery (deterministic because ALLOWED_EXTS is ordered)
    discovered = []

    for ext in ALLOWED_EXTS:
        p = img_dir / f"{stem}.{ext}"
        if p.exists():
            try:
                real = p.resolve()
            except Exception:
                # if resolve() fails for any reason, fallback to the path itself
                real = p
            if real not in resolved_map:
                resolved_map[real] = p
                discovered.append((real, p))

    if not discovered:
        return None, []

    # If more than one distinct real file exists (rare), choose based on PREFERRED_ORDER.
    if len(discovered) == 1:
        chosen_real = discovered[0][0]  # the resolved Path
        # prepare user-visible list of filenames (use the actual on-disk name)
        visible_matches = [d[0].name for d in discovered]
        return chosen_real, visible_matches

    # multiple distinct real files found (different files with same stem but different suffix)
    # choose the one whose suffix matches PREFERRED_ORDER first
    for pref in PREFERRED_ORDER:
        for real, orig in discovered:
            if real.suffix.lower().lstrip('.') == pref.lower().lstrip('.'):
                visible_matches = [d[0].name for d in discovered]
                return real, visible_matches

    # fallback: return first discovered real file
    chosen_real = discovered[0][0]
    visible_matches = [d[0].name for d in discovered]
    return chosen_real, visible_matches

def get_correct_index_from_json(obj):
    if "correct_index" in obj and obj["correct_index"] is not None:
        return int(obj["correct_index"])
    if "original_answer_letter" in obj and obj.get("options"):
        letter = obj["original_answer_letter"]
        if isinstance(letter, str) and len(letter) >= 1:
            mapping = { 'A':0,'B':1,'C':2,'D':3,'E':4 }
            L = letter.strip()[0].upper()
            if L in mapping:
                return mapping[L]
    if "correct" in obj and isinstance(obj["correct"], int):
        return int(obj["correct"])
    if "answer_index" in obj:
        return int(obj["answer_index"])
    return None

def build_question_xml_with_box(qname, image_filename, image_b64, left, top, width, height, qtext_static, marker_label):
    """
    Build question XML including <targetbox> metadata for the square.
    Keep similar structure to previous script; the <targetbox> is included for plugin metadata.
    """
    qname_esc = html.escape(qname)
    qtext_html = f"""<p>{html.escape(qtext_static)}</p>
<img src="@@PLUGINFILE@@/{html.escape(image_filename)}" alt="{html.escape(qname)}" />"""
    xml_parts = []
    xml_parts.append('  <question type="ddmarker">')
    xml_parts.append(f'    <name><text>{qname_esc}</text></name>')
    xml_parts.append('    <questiontext format="html">')
    xml_parts.append(f'      <text><![CDATA[{qtext_html}]]></text>')
    xml_parts.append('    </questiontext>')
    xml_parts.append('    <generalfeedback format="html"><text></text></generalfeedback>')
    xml_parts.append('    <defaultgrade>1.0000000</defaultgrade>')
    xml_parts.append('    <penalty>0.0000000</penalty>')
    xml_parts.append('    <hidden>0</hidden>')
    # Emit drag/drop in Moodle ddmarker export format (one draggable, one drop zone)
    # draggable definition
    # --- Emit image file (closed) before options ---
    xml_parts.append(f'    <file name="{html.escape(image_filename)}" path="/" encoding="base64">')
    xml_parts.append(image_b64)
    xml_parts.append('    </file>')

    # --- Emit options containing drag(s) and drop zone(s) ---
    # compute coords robustly: x1,y1;x2,y2 (top-left;bottom-right)
    try:
        x1 = int(left)
        y1 = int(top)
    except Exception:
        x1 = 0; y1 = 0
    # prefer size if present (square), else right/bottom
    try:
        if 'size' in locals() and size is not None:
            x2 = x1 + int(width); y2 = y1 + int(width)
        else:
            x2 = int(right); y2 = int(bottom)
    except Exception:
        x2 = x1; y2 = y1
    # ensure ordering and clamp to image dims if available
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    if 'img_w' in locals() and img_w is not None:
        x1 = max(0, min(img_w, x1)); x2 = max(0, min(img_w, x2))
    if 'img_h' in locals() and img_h is not None:
        y1 = max(0, min(img_h, y1)); y2 = max(0, min(img_h, y2))
    # convert to x,y;width,height (width,height) per Moodle JSON expectation
    w = int(x2 - x1)
    h = int(y2 - y1)
    # Interpret left/top/width/height and produce Moodle coords "x,y;w,h"
    try:
        x = int(left)
        y = int(top)
    except Exception:
        x = 0
        y = 0
    try:
        w = int(width)
        h = int(height)
    except Exception:
        # fallback: handle older 'size' (square), or right-bottom
        if 'size' in locals() and size is not None:
            w = int(width); h = int(width)
        else:
            try:
                w = int(right) - int(left)
                h = int(bottom) - int(top)
            except Exception:
                w = 0; h = 0

    # guard against negative sizes
    if w < 0: w = abs(w)
    if h < 0: h = abs(h)

    # clamp to image dimensions if available
    try:
        if 'img_w' in locals() and img_w is not None:
            x = max(0, min(img_w - 1, x))
            w = min(w, max(0, img_w - x))
        if 'img_h' in locals() and img_h is not None:
            y = max(0, min(img_h - 1, y))
            h = min(h, max(0, img_h - y))
    except Exception:
        pass

    coords = f"{x},{y};{w},{h}"

    xml_parts.append('      <drag>')
    xml_parts.append('        <no>1</no>')
    xml_parts.append(f'        <text>{marker_label}</text>')
    xml_parts.append('        <noofdrags>1</noofdrags>')
    xml_parts.append('      </drag>')
    xml_parts.append('      <drop>')
    xml_parts.append('        <no>1</no>')
    xml_parts.append('        <shape>rectangle</shape>')
    xml_parts.append(f'        <coords>{coords}</coords>')
    xml_parts.append('        <choice>1</choice>')
    xml_parts.append('      </drop>')

    xml_parts.append('  </question>')
    return "\n".join(xml_parts)
    return "\n".join(xml_parts)


def main():
    p = argparse.ArgumentParser(description="Generate Moodle XML package with square targets from plate JSON + images")
    p.add_argument("--json-dir", required=True, help="Directory with per-plate JSON files")
    p.add_argument("--img-dir", required=False, help="Directory with plate PNG images (default: same as json_dir)")
    p.add_argument("--out", required=False, default="moodle_square.zip", help="Output zip filename")
    p.add_argument("--pad", type=int, default=0, help="Optional padding (px) to add around the correct box before making the square")
    p.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = p.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        print("ERROR: json-dir not found:", json_dir)
        sys.exit(2)
    img_dir = Path(args.img_dir) if args.img_dir else json_dir

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found in", json_dir)
        sys.exit(1)

    static_qtext = 'Please drag the "Correct Answer" marker below the image to the text of the answer that matches the best answer. Make sure that top left target marker is in the correct location.'
    marker_label = "Drag Me (Top-Left Marker counts)"

    qblocks = []
    images = {}
    csv_rows = []
    name_counter = Counter()
    skipped = 0

    for jf in json_files:
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: failed to parse JSON {jf}: {e}")
            skipped += 1
            continue

        question_text = obj.get("question") or obj.get("Question") or ""
        qname_base = sanitize_name(question_text) if question_text else jf.stem
        name_counter[qname_base] += 1
        qname = qname_base if name_counter[qname_base] == 1 else f"{qname_base}_{name_counter[qname_base]}"

        boxes = obj.get("boxes") or obj.get("option_boxes") or obj.get("boxes_px")
        if not boxes or not isinstance(boxes, (list,tuple)):
            print(f"WARNING: {jf.name} missing 'boxes'; skipping.")
            skipped += 1
            continue

        correct_idx = get_correct_index_from_json(obj)
        if correct_idx is None or correct_idx < 0 or correct_idx >= len(boxes):
            print(f"WARNING: {jf.name} missing/invalid correct index; skipping.")
            skipped += 1
            continue

        box = boxes[correct_idx]
        if not (isinstance(box, (list,tuple)) and len(box) >= 4):
            print(f"WARNING: {jf.name} invalid box format {box}; skipping.")
            skipped += 1
            continue

        # determine image file to use
        image_path, matches = find_image_for_json_strict(jf, img_dir)
        if image_path is None:
            print(f"WARNING: No image found for {jf.name} (expected {jf.stem}.<ext>); skipping.")
            skipped += 1
            continue
        if len(matches) > 1:
            print(f"WARNING: Multiple distinct image files found for {jf.name}: {matches}. Using {image_path.name}.")
        # image_path is a resolved Path -> use image_name = image_path.name next
        image_name = image_path.name

        # read image dimensions for boundary clipping
        try:
            from PIL import Image
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except Exception:
            # fallback to no-clamp sizes; we'll assume 700x300 as default
            img_w, img_h = 700, 300

        left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # use rectangle coordinates directly (width = right-left, height = bottom-top)
        width = max(0, int(right) - int(left))
        height = max(0, int(bottom) - int(top))
        
        image_b64 = embed_file_base64(image_path)
        qxml = build_question_xml_with_box(qname, image_name, image_b64, left, top, width, height, static_qtext, marker_label)

        qblocks.append(qxml)
        images[image_name] = image_path
        # append a fresh dict per-row to avoid reuse of the same mutable object
        csv_rows.append({
            'image': image_name,
            'json_file': jf.name,
            'qname': qname,
            'correct_index': int(correct_idx) if correct_idx is not None else '',
            'orig_left': int(left), 'orig_top': int(top), 'orig_right': int(right), 'orig_bottom': int(bottom),
            'rect_left': int(left), 'rect_top': int(top), 'rect_width': int(width), 'rect_height': int(height), 'rect_right': int(left + width), 'rect_bottom': int(top + height)
        })

        if not args.quiet:
            print(f"Added {jf.name} -> image {image_name}  box L,T,S = {rect_left},{rect_top},{rect_width},{rect_height}")

    if not qblocks:
        print("No questions generated; exiting.")
        sys.exit(1)

    header = '<?xml version="1.0" encoding="UTF-8"?>\n<quiz>\n'
    footer = '\n</quiz>\n'
    xml_all = header + "\n".join(qblocks) + footer

    xml_out = "questions.xml"
    with open(xml_out, "w", encoding="utf-8") as fh:
        fh.write(xml_all)
    print("Wrote", xml_out)

    csv_out = "plates_moodle_coords_square.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        csv_fieldnames = [
            'image','json_file','qname','correct_index',
            'orig_left','orig_top','orig_right','orig_bottom',
            'rect_left','rect_top','rect_width','rect_height','rect_bottom'
        ]
        w = csv.DictWriter(fh, fieldnames=csv_fieldnames)
        w.writeheader()
        for r in csv_rows:
            # ensure all fields exist
            for fn in csv_fieldnames:
                if fn not in r:
                    r[fn] = ''
            w.writerow(r)
    print("Wrote", csv_out)

    import zipfile
    outzip = args.out
    with zipfile.ZipFile(outzip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_out, arcname=xml_out)
        for name, path in images.items():
            zf.write(str(path), arcname=name)
        zf.write(csv_out, arcname=csv_out)
    print("Wrote Moodle package:", outzip)
    print(f"Skipped {skipped} plates.")
    print("If Moodle import ignores <targetbox>, use the CSV to paste coordinates manually or tell me the import error so I can adapt the XML schema.")
    print("Done.")

if __name__ == "__main__":
    main()
