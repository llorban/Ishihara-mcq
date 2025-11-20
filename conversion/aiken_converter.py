#!/usr/bin/env python3
"""
aiken_to_json_set.py

Parse an AIKEN-style file and write one JSON file per question.

Usage:
  python3 aiken_to_json_set.py --in "quiz2 Aggression.txt" --out json_out --prefix plate_

Output files: outdir/<prefix>01.json, <prefix>02.json, ...
Each JSON contains:
  id, question, options (list), correct_index (0-based), original_answer_letter, source_file
"""

import argparse, json, os, re, sys
from pathlib import Path

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}

def parse_aiken_text(text):
    """
    Parse AIKEN text into a list of dicts: {question, options(list len4), answer_letter}
    This is permissive: it finds lines, groups by blank-line or by "ANSWER:" marker.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    items = []
    idx = 0
    n = len(lines)
    while idx < n:
        # skip leading empty lines
        while idx < n and lines[idx].strip() == "":
            idx += 1
        if idx >= n:
            break
        # read question line(s) until we hit an option line starting with "A." or "A)"
        q_lines = []
        while idx < n and not re.match(r'^[Aa][\.\)]\s+', lines[idx]) and not re.match(r'^\s*ANSWER\s*:', lines[idx], flags=re.I):
            q_lines.append(lines[idx])
            idx += 1
        if not q_lines:
            # nothing identified as question, bail
            break
        question = " ".join([q.strip() for q in q_lines]).strip()

        # read options A-D
        opts = []
        for expected in ['A','B','C','D']:
            # skip blank lines
            while idx < n and lines[idx].strip() == "":
                idx += 1
            if idx >= n:
                break
            m = re.match(r'^\s*([A-Za-z])[\.\)]\s*(.*)$', lines[idx])
            if m:
                letter = m.group(1).upper()
                text_opt = m.group(2).strip()
                opts.append((letter, text_opt))
                idx += 1
            else:
                # If not in A. form, perhaps options are on separate lines without labels; try to accept plain lines
                # But for strict AIKEN we require A./A) format; break
                break

        # ensure we have at least 1 option; normalize to exactly 4 if possible
        if len(opts) < 4:
            # try to continue scanning until we find ANSWER:
            # collect lines until we reach ANSWER:
            while idx < n and not re.match(r'^\s*ANSWER\s*:', lines[idx], flags=re.I):
                if lines[idx].strip() != "":
                    # attempt to parse any unlabeled option
                    opts.append((None, lines[idx].strip()))
                idx += 1

        # now find ANSWER:
        ans_letter = None
        while idx < n and lines[idx].strip() == "":
            idx += 1
        if idx < n:
            m_ans = re.match(r'^\s*ANSWER\s*:\s*([A-Za-z0-9])', lines[idx], flags=re.I)
            if m_ans:
                ans_letter = m_ans.group(1).upper()
                idx += 1
            else:
                # possibly answer inline on same line as others; search forward a few lines
                j = idx
                found = False
                while j < min(n, idx+6):
                    m2 = re.match(r'^\s*ANSWER\s*:\s*([A-Za-z0-9])', lines[j], flags=re.I)
                    if m2:
                        ans_letter = m2.group(1).upper()
                        idx = j+1
                        found = True
                        break
                    j += 1
                if not found:
                    # no answer found — we'll record None and continue
                    ans_letter = None

        # assemble options texts in the correct A,B,C,D order if labeled
        # if opts have labels, sort by label; else take first 4
        opts_sorted = None
        if all(letter is not None for (letter, _) in opts[:4]):
            # create dict letter->text
            mapping = {}
            for letter,text in opts:
                mapping[letter.upper()] = text
            # try to form A-D list
            try:
                opts_sorted = [mapping['A'], mapping['B'], mapping['C'], mapping['D']]
            except KeyError:
                # fallback: use order encountered
                opts_sorted = [t for (_,t) in opts][:4]
        else:
            opts_sorted = [t for (_,t) in opts][:4]

        items.append({"question": question, "options": opts_sorted, "answer_letter": ans_letter})
    return items

def write_json_files(items, outdir, prefix="plate_",
                     source_file_name=None):
    os.makedirs(outdir, exist_ok=True)
    written = 0
    for i, it in enumerate(items, start=1):
        # validation
        q = it.get("question", "").strip()
        opts = it.get("options") or []
        if not q:
            print(f"WARNING: item {i} missing question; skipping.")
            continue
        if len(opts) < 2:
            print(f"WARNING: item {i} has <2 options; skipping. Q: {q[:40]}")
            continue
        # ensure we have exactly 4 options (pad with empty strings if necessary)
        if len(opts) < 4:
            opts = opts + [""] * (4 - len(opts))
        else:
            opts = opts[:4]
        ans = it.get("answer_letter")
        correct_index = None
        if ans and ans.upper() in LETTER_TO_INDEX:
            correct_index = LETTER_TO_INDEX[ans.upper()]
        # build meta
        ident = f"{prefix}{i:02d}"
        obj = {
            "id": ident,
            "question": q,
            "options": opts,
            "correct_index": correct_index,
            "original_answer_letter": ans,
            "source_file": source_file_name
        }
        outpath = os.path.join(outdir, f"{ident}.json")
        with open(outpath, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
        written += 1
    return written

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="AIKEN input file path")
    p.add_argument("--out", dest="outdir", default="json_out", help="Output directory")
    p.add_argument("--prefix", dest="prefix", default="plate_", help="Filename prefix for JSONs")
    args = p.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        print("Input file not found:", infile)
        sys.exit(1)
    txt = infile.read_text(encoding="utf-8")
    items = parse_aiken_text(txt)
    if not items:
        print("No items parsed; check your AIKEN file format.")
        sys.exit(1)

    written = write_json_files(items, args.outdir, prefix=args.prefix, source_file_name=infile.name)
    print(f"Wrote {written} JSON files to {args.outdir}")

if __name__ == "__main__":
    main()
