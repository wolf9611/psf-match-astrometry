#!/usr/bin/env bash
# =============================================================================
# generate_all_artifacts.sh
#
# Runs the PSF-match pipeline with every meaningful CLI-flag combination on
# both supported instruments (DECam and FORS2), generating the full matrix of
# output artifacts under output/<instrument>/<variant>/.
#
# Each run is independent (writes to a distinct subdirectory).  The script
# executes them sequentially to avoid concurrent Gaia-archive query
# conflicts.
#
# Usage:
#   bash generate_all_artifacts.sh
#
# Images and Gaia CSVs are not in the repository. Place your own files at
# the paths in src/config.py (see README, "Obtaining public data") before
# running this script.
#
# Estimated wall-clock time on a typical workstation: 2–3 hours.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================="
echo " PSF-match — generate all artifacts"
echo "============================================="
echo ""

# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------
run() {
    local label="$1"
    shift
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [${label}] python main.py $*"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python main.py "$@"
    echo "  ✓  ${label}  completed"
}

# ====================================================================
# DECam  (default instrument)
# ====================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    DECam  pipeline runs                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

run "DECam faint+pm_damped"     --no-clean
run "DECam nofaint+pm_damped"   --no-clean --no-faint-stars
run "DECam faint+pm_full"       --no-clean --pm-mode full
run "DECam faint+pm_none"       --no-clean --pm-mode none
run "DECam compare-faint-stars" --no-clean --compare-faint-stars

# ====================================================================
# FORS2  (--preset FORS2)
# ====================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   FORS2  pipeline runs                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

run "FORS2 faint+pm_damped"     --no-clean --preset FORS2
run "FORS2 nofaint+pm_damped"   --no-clean --preset FORS2 --no-faint-stars
run "FORS2 faint+pm_full"       --no-clean --preset FORS2 --pm-mode full
run "FORS2 faint+pm_none"       --no-clean --preset FORS2 --pm-mode none
run "FORS2 compare-faint-stars" --no-clean --preset FORS2 --compare-faint-stars

echo ""
echo "============================================="
echo " All pipeline runs completed successfully."
echo "============================================="
