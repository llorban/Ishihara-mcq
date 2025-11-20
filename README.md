# Ishihara-mcq
Multiple Choice Questions Strike Back

## What is this?
A program that will convert AIKEN formatted multiple-choice test items into distorted image based label dragging questions. 

## Components
### conversion/aiken_converter.py
Taken an AIKEN formatted file of MCQ items and converts it to JSON files

### mcq_gen10.py
Takes a JSON file from json_converter/* and generates images saved in plates_out/ 

### moodlexml_gen11.py
Takes images from plates_out/* and generates a Moodle XML file label dragging questions.

## Example
### Generate JSON
- python3.13 conversion/aiken_converter.py --in input/t1_wk2_2400/MCQ\ Week\ 2\ Measurement\ Variables\ Constructs.txt --out json_converter --prefix "t1_2400"

### Generate Ishihara Image
- python3.13 mcq_gen10.py --json-dir json_converter --out plates_out/t1_2400

### Generate Moodle XML
- python3.13 moodlexml_gen11.py --json-dir plates_out/t3_2400/ch9 --img-dir plates_out/t3_2400/ch9 --out moodle_xml/t3_2400_ch9.zip --pad 4

### Find the xml file and import into your Moodle instance
