# LTRC Assignment 2: forced alignment and multimodal emotion recognition

**Report (both parts, one PDF):** [`Report/Report.pdf`](Report/Report.pdf). Part A covers 2a and part B covers 2b.

| Part | Folder | Report source |
|---|---|---|
| 2a: forced alignment of 6 English recordings with the Montreal Forced Aligner | [`2a_forced_alignment/`](2a_forced_alignment/) | `2a_forced_alignment/Report/Report_2a.md` |
| 2b: emotion recognition on TESS from speech, text and both | [`2b_emotion_recognition/`](2b_emotion_recognition/) | `2b_emotion_recognition/Report/Report_2b.md` |

Each folder has its own README with install and run instructions.

Rebuild the PDF (pandoc + headless Chrome; macOS Chrome path):

```bash
python3 tools/combine_reports.py && tools/build_pdf.sh Report/Report.md
```
