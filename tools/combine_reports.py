"""Both part reports (2a_*/Report/Report_2a.md, 2b_*/Report/Report_2b.md) -> one submission report, Report/Report.md.
Headings move one level down and get the part letter (A = 2a, B = 2b), as do the § cross-references; image paths are
rewritten for the new location. Build the PDF with: python3 tools/combine_reports.py && tools/build_pdf.sh Report/Report.md
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/sudarshan032/ltrc-assignment2-alignment-emotion"
PARTS = [("A", "2a_forced_alignment", "Report_2a.md"), ("B", "2b_emotion_recognition", "Report_2b.md")]


def part(letter, folder, name):
    text = (ROOT / folder / "Report" / name).read_text()
    meta, body = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S).groups()
    title = re.search(r'^title: "(.*)"', meta, re.M).group(1)
    body = re.sub(r"\*\*Repository:\*\* <[^>]+> \(folder `[^`]+`: [^)]*\)\.? ?", "", body)  # folders named once, at the top
    body = re.sub(r"^(#+) (\d+(?:\.\d+)*\.?) ", lambda m: f"#{m[1]} {letter}{m[2]} ", body, flags=re.M)
    body = re.sub(r"§(\d)", rf"§{letter}\1", body)
    body = body.replace("](../", f"](../{folder}/")
    return f"# Part {letter} ({folder[:2]}): {title}\n\n" + body.strip() + "\n"


def main():
    head = ('---\ntitle: "Assignment 2: Forced Alignment and Multimodal Emotion Recognition"\n'
            'subtitle: "LTRC, IIIT Hyderabad"\nauthor: "G Sudarshan Sastry"\ndate: "November 2026"\n---\n\n'
            f"**Repository:** <{REPO}>. Part A (2a) is in `2a_forced_alignment/`, part B (2b) in `2b_emotion_recognition/`.\n\n")
    out = head + '\n<div style="page-break-before: always"></div>\n\n'.join(part(*p) for p in PARTS)
    (ROOT / "Report").mkdir(exist_ok=True)
    (ROOT / "Report" / "Report.md").write_text(out)


if __name__ == "__main__":
    main()
