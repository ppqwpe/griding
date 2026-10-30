#!/bin/bash
# Griding for macOS, CPU only - use this if Core ML acceleration causes trouble. Everything it needs is inside this folder.
# First time only: macOS may block it because it came from the internet.
#   Right-click Griding.command > Open > Open   (or System Settings > Privacy & Security > Open Anyway)
cd "$(dirname "$0")" || exit 1
PY="app/python-mac/bin/python3.12"
if [ ! -x "$PY" ]; then
  echo
  echo "  Griding files are missing next to this file."
  echo "  Unzip the whole zip first, then run Griding.command in the unzipped folder."
  echo
  read -r -p "  Press Return to close. "
  exit 1
fi
# Files unzipped from a download carry a 'quarantine' mark that stops the bundled Python
# libraries from loading. Clear it for this folder only (nothing outside is touched).
xattr -dr com.apple.quarantine . 2>/dev/null
export PYTHONPATH="$PWD/app/lib-mac"
export PYTHONNOUSERSITE=1
export FACEGRID_CPU=1
"$PY" -u -X utf8 app/server.py "$@"
code=$?
if [ $code -ne 0 ]; then
  echo
  read -r -p "  Griding stopped (code $code). Press Return to close. "
fi
exit $code
