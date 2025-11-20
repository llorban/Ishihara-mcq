#!/usr/bin/env python3
"""
plates_to_moodle_xml.py

Generate a Moodle XML package (zip) for drag-and-drop marker questions from
plates images + per-plate JSON files.

Usage:
  python3 plates_to_moodle_xml.py --json-dir ./json_out --img-dir ./plates_png --out moodle_package.zip

Outputs:
  - questions.xml inside the zip
  - the PNG image files referenced (embedded)
  - plates_moodle_coords.csv in the working directory listing centers/radii

Notes:
 - The JSON format expected per file (one-per-plate) is:
   {
     "id": "plate_01",
     "question": "Which pattern ... ?",
     "options": [...],
     "correct_index": 0,
     "boxes": [[left,top,right,bottom], ...]
   }
 - If correct_index is missing, script will attempt to use "original_answer_letter" or "correct" keys if present.
 - If no correct target can be inferred, the plate will be skipped (warning).
 - If your Moodle import rejects the XML, save the CSV and tell me the error — I will adjust the XML schema accordingly.
"""

import os, sys, argparse, json, base64, glob, csv, html
from pathlib import Path
from collections import Counter

def sanitize_name(s):
    # take first few (alpha/numeric) words, strip punctuation
    import re
    tokens = re.findall(r"[A-Za-z0-9']+", s)
    if not tokens:
        return "plate"
    name = "_".join(tokens[:7])
    return name[:60]

def compute_center_and_radius(box):
    # box = [left, top, right, bottom]
    l,t,r,b = box
    cx = int(round((l + r) / 2.0))
    cy = int(round((t + b) / 2.0))
    w = max(1, int(r - l))
    h = max(1, int(b - t))
    # radius: half the shorter dimension, but clamp to reasonable range
    r0 = max(12, min(max(4, min(w, h)//2), max(w,h)//2))
    return cx, cy, r0

def get_correct_index_from_json(obj):
    if "correct_index" in obj and obj["correct_index"] is not None:
        return int(obj["correct_index"])
    # try alternative keys
    if "original_answer_letter" in obj and obj.get("options"):
        letter = obj["original_answer_letter"]
        if isinstance(letter, str) and len(letter) >= 1:
            mapping = { 'A':0,'B':1,'C':2,'D':3,'E':4 }
            L = letter.strip()[0].upper()
            if L in mapping:
                return mapping[L]
    if "correct" in obj and isinstance(obj["correct"], int):
        return int(obj["correct"])
    return None

def embed_file_base64(path):
    with open(path, "rb") as fh:
        data = fh.read()
    b64 = base64.b64encode(data).decode("ascii")
    return b64

def build_question_xml(qname, image_filename, image_b64, center_x, center_y, radius_px, qtext_static, marker_label):
    """
    Build a single <question type="ddmarker"> ... </question> block.
    The exact ddmarker schema varies across Moodle versions; this template uses a commonly accepted structure:
      - refs image with @@PLUGINFILE@@ in question HTML
      - includes the image file as a <file> child (base64)
      - adds a small <dragmarker> metadata block containing coords (for ease of manual check)
    If import complains, we can adapt quickly — keep the CSV produced by this script for fallback.
    """
    # escape XML/HTML where needed
    qname_esc = html.escape(qname)
    qtext_html = f"""<p>{html.escape(qtext_static)}</p>
<img src="@@PLUGINFILE@@/{html.escape(image_filename)}" alt="{html.escape(qname)}" />"""
    # Build question block
    qxml = []
    qxml.append(f'  <question type="ddmarker">')
    qxml.append(f'    <name><text>{qname_esc}</text></name>')
    qxml.append(f'    <questiontext format="html">')
    qxml.append(f'      <text><![CDATA[{qtext_html}]]></text>')
    qxml.append(f'    </questiontext>')
    qxml.append(f'    <generalfeedback format="html"><text></text></generalfeedback>')
    qxml.append(f'    <defaultgrade>1.0000000</defaultgrade>')
    qxml.append(f'    <penalty>0.0000000</penalty>')
    qxml.append(f'    <hidden>0</hidden>')
    # plugin-specific payload: marker label and position included as metadata (not guaranteed to be read by all plugins)
    # but many plugin imports accept custom XML tags; this gives a clear starting point for adjustments
    qxml.append(f'    <dragmarker>')
    qxml.append(f'      <markerlabel>{html.escape(marker_label)}</markerlabel>')
    qxml.append(f'      <markerx>{center_x}</markerx>')
    qxml.append(f'      <markery>{center_y}</markery>')
    qxml.append(f'      <markerradius>{radius_px}</markerradius>')
    qxml.append(f'      <correct>1</correct>')
    qxml.append(f'    </dragmarker>')
    # include the file data so the package is self-contained
    qxml.append(f'    <file name="{html.escape(image_filename)}" path="/" encoding="base64">')
    qxml.append(image_b64)
    qxml.append(f'    </file>')
    qxml.append(f'  </question>')
    return "\n".join(qxml)

def main():
    p = argparse.ArgumentParser(description="Generate Moodle XML package (zip) from plates JSON + images")
    p.add_argument("--json-dir", required=True, help="Directory with per-plate JSON files")
    p.add_argument("--img-dir", required=False, help="Directory with plate PNG images (default: same as json_dir)")
    p.add_argument("--out", required=False, default="moodle_ddmarker.zip", help="Output zip filename")
    p.add_argument("--radius", type=int, default=None, help="Optional fixed radius in px to use for all markers (overrides computed radius)")
    p.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = p.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        print("ERROR: json-dir not found:", json_dir)
        sys.exit(2)
    img_dir = Path(args.img_dir) if args.img_dir else json_dir

    # find JSON files (sorted)
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found in", json_dir)
        sys.exit(1)

    # collect questions & images
    question_blocks = []
    image_files_to_zip = {}
    coord_rows = []
    name_counter = Counter()
    skipped = 0

    static_qtext = 'Please drag the "Correct Answer" marker below the image to the text of the answer that matches the best answer. Make sure that top left target marker is in the correct location.'
    marker_label = "Drag Me (Top-Left Marker counts)"

    for idx, jf in enumerate(json_files, start=1):
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: failed to parse JSON {jf}: {e}")
            skipped += 1
            continue

        question_text = obj.get("question") or obj.get("Question") or ""
        qname_base = sanitize_name(question_text) if question_text else jf.stem
        # ensure uniqueness
        name_counter[qname_base] += 1
        qname = qname_base if name_counter[qname_base] == 1 else f"{qname_base}_{name_counter[qname_base]}"

        # Try to find boxes
        boxes = obj.get("boxes") or obj.get("option_boxes") or obj.get("boxes_px") or obj.get("boxes_pixels")
        if not boxes or not isinstance(boxes, (list,tuple)):
            print(f"WARNING: JSON {jf.name} missing 'boxes' array; skipping.")
            skipped += 1
            continue

        # determine correct_index
        correct_idx = get_correct_index_from_json(obj)
        if correct_idx is None:
            # fallback: if 'correct_answer' or 'answer_index' present
            if "answer_index" in obj:
                correct_idx = int(obj["answer_index"])
            else:
                print(f"WARNING: {jf.name} missing correct_index; skipping.")
                skipped += 1
                continue

        if correct_idx < 0 or correct_idx >= len(boxes):
            print(f"WARNING: {jf.name} correct_index out of range; skipping.")
            skipped += 1
            continue

        # get center/radius
        box = boxes[correct_idx]
        try:
            cx, cy, rad = compute_center_and_radius(box)
        except Exception as e:
            print(f"WARNING: {jf.name} invalid box data {box}: {e}; skipping.")
            skipped += 1
            continue

        if args.radius is not None:
            rad = int(args.radius)

        # find corresponding image (prefer id + .png)
        # candidate names: jf.stem+'.png', obj.get('id')+'.png', jf.stem.replace('json','png')
        candidates = []
        if (img_dir / (jf.stem + ".png")).exists():
            candidates.append(img_dir / (jf.stem + ".png"))
        if obj.get("id") and (img_dir / (str(obj.get("id")) + ".png")).exists():
            candidates.append(img_dir / (str(obj.get("id")) + ".png"))
        # fallback: any png with same prefix
        pref = jf.stem
        for f in img_dir.glob(pref + "*.png"):
            candidates.append(f)
        # fallback full directory search if only one png exists
        if not candidates:
            all_pngs = list(img_dir.glob("*.png"))
            if len(all_pngs) == 1:
                candidates.append(all_pngs[0])

        if not candidates:
            print(f"WARNING: No image PNG found for {jf.name}; skipping.")
            skipped += 1
            continue

        image_path = candidates[0]
        image_name = image_path.name
        image_b64 = embed_file_base64(image_path)
        # build question xml block
        qxml = build_question_xml(qname, image_name, image_b64, cx, cy, rad, static_qtext, marker_label)
        question_blocks.append(qxml)
        image_files_to_zip[image_name] = image_path
        coord_rows.append([image_name, jf.name, qname, correct_idx, cx, cy, rad])

        if not args.quiet:
            print(f"Added: {jf.name} -> image {image_name}  center=({cx},{cy}) r={rad}")

    if not question_blocks:
        print("No questions generated; exiting.")
        sys.exit(1)

    # build full XML (quiz)
    header = '<?xml version="1.0" encoding="UTF-8"?>\n<quiz>\n'
    footer = '\n</quiz>\n'
    body = "\n".join(question_blocks)
    xml_full = header + body + footer

    # write standalone xml for inspection
    xml_out = "questions.xml"
    with open(xml_out, "w", encoding="utf-8") as fh:
        fh.write(xml_full)
    print("Wrote", xml_out)

    # write CSV of coords
    csv_out = "plates_moodle_coords.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image","json_file","qname","correct_index","center_x","center_y","radius"])
        w.writerows(coord_rows)
    print("Wrote", csv_out)

    # create ZIP package
    import zipfile
    outzip = args.out
    with zipfile.ZipFile(outzip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # add xml
        zf.write(xml_out, arcname=xml_out)
        # add images
        for iname, ipath in image_files_to_zip.items():
            zf.write(str(ipath), arcname=iname)
        # include the CSV as well
        zf.write(csv_out, arcname=csv_out)
    print("Wrote Moodle package:", outzip)
    print(f"Skipped {skipped} plates due to missing data or images.")
    print("IMPORTANT: If Moodle import rejects the package, send me the import error and I'll adapt the XML schema quickly. You can also use the CSV to manually paste coordinates into Moodle's UI.")
    print("Done.")

if __name__ == "__main__":
    main()
