# validator.py
import zipfile
import xml.etree.ElementTree as ET
import re
import sys

def validate_coords_string(s):
    # accept "x1,y1;x2,y2" with ints
    m = re.match(r'^\s*([+-]?\d+),([+-]?\d+);([+-]?\d+),([+-]?\d+)\s*$', s)
    if not m:
        return False, "coords format invalid"
    x1,y1,x2,y2 = map(int, m.groups())
    if x2 < x1 or y2 < y1:
        return False, f"inverted coords ({x1},{y1};{x2},{y2})"
    return True, (x1,y1,x2,y2)

def validate_moodle_xml_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        if "moodle.xml" in z.namelist():
            data = z.read("moodle.xml")
        else:
            # attempt to find first .xml
            xmlfiles = [n for n in z.namelist() if n.lower().endswith(".xml")]
            if not xmlfiles:
                print("No XML file found in ZIP")
                return False
            data = z.read(xmlfiles[0])
    # parse
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        print("XML parse error:", e)
        return False

    # find ddmarker questions
    ok = True
    for q in root.findall(".//question"):
        if q.get("type") == "ddmarker":
            # check file present before options
            file_el = q.find("file")
            options_el = q.find("options")
            if file_el is None:
                print("Missing <file> element for question:", ET.tostring(q.find('name'), encoding='unicode'))
                ok = False
            if options_el is None:
                print("Missing <options> element for question:", ET.tostring(q.find('name'), encoding='unicode'))
                ok = False
            # check drop coords
            for drop in q.findall(".//drop"):
                coords = drop.findtext("coords") or ""
                v, info = validate_coords_string(coords)
                if not v:
                    print("Bad coords for question", q.findtext("name/text"), "coords:", coords, "->", info)
                    ok = False
    if ok:
        print("Basic validation OK: XML well-formed and ddmarker sections present.")
    else:
        print("Validation found issues.")
    return ok

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validator.py package.zip")
        sys.exit(2)
    validate_moodle_xml_from_zip(sys.argv[1])
