#!/usr/bin/env bash
# Markdown report -> standalone HTML (pandoc) -> PDF (headless Chrome).
# Usage: tools/build_pdf.sh <report.md>      (writes <report>.pdf next to it)
set -euo pipefail
md=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
html=${md%.md}.html; pdf=${md%.md}.pdf
here=$(cd "$(dirname "$0")" && pwd)
(cd "$(dirname "$md")" && pandoc "$md" -s --embed-resources --css "$here/report.css" -o "$html")
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$pdf" "file://$html" 2>/dev/null
rm "$html"
echo "$pdf"
