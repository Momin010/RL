#!/usr/bin/env bash
# End-to-end: train both networks (stabilization + evasion), export weights to
# C, then build and run the host parity tests that prove the C++ inference
# matches the Python models.
#
# Usage:  bash tools/run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> [1/6] Train stabilization net (real F-motor curves, BC+DAgger+RL)"
python3 scripts/train_stab.py

echo "==> [2/6] Train evasion net"
python3 scripts/train.py

echo "==> [3/6] Export stabilization weights to C header"
python3 scripts/export_stab_c.py

echo "==> [4/6] Export evasion weights to C header"
python3 scripts/export_c.py

echo "==> [5/6] Build + run stabilization parity test"
TMP="$(mktemp -d)"
g++ -O2 -std=c++14 -Wall -Wextra -I firmware firmware/host_stab_parity_test.cpp -o "$TMP/parity_stab"
"$TMP/parity_stab"

echo "==> [6/6] Build + run evasion parity test"
g++ -O2 -std=c++14 -Wall -Wextra -I firmware firmware/host_parity_test.cpp -o "$TMP/parity"
"$TMP/parity"

echo "==> Done. Flash firmware/rocket_evasion.ino to the Teensy 4.1."
