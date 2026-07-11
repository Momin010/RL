#!/usr/bin/env bash
# End-to-end: train the network, export weights to C, then build and run the
# host parity test that proves the C++ inference matches the Python model.
#
# Usage:  bash tools/run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> [1/3] Train"
python3 scripts/train.py

echo "==> [2/3] Export weights to C header"
python3 scripts/export_c.py

echo "==> [3/3] Build + run host parity test"
OUT="$(mktemp -d)/parity"
g++ -O2 -std=c++14 -Wall -Wextra -I firmware firmware/host_parity_test.cpp -o "$OUT"
"$OUT"

echo "==> Done. Flash firmware/rocket_evasion.ino to the Teensy 4.1."
