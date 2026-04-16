#!/usr/bin/env bash
# Full pipeline: download → DEM → preprocess → (repeat per region) → combine → train → visualize
# Deletes raw .nc files after each region to keep disk usage low.

MAX_FILES=200      # granules per region (~10 years of 12-day repeat)
MAX_SAMPLES=1000   # pixels sampled per region during preprocessing

set -euo pipefail
cd "$(dirname "$0")"

REGIONS=(
    # tectonic
    san_andreas hayward garlock imperial san_jacinto owens_valley
    # landslide
    oregon_coast california_north washington slumgullion grand_mesa oso_washington portuguese_bend la_conchita thistle_utah
    # subsidence
    san_joaquin houston phoenix las_vegas permian_basin stockton_delta tucson
    # stable
    stable_sierra stable_texas stable_kansas stable_nebraska stable_iowa stable_wyoming stable_colorado
)

log() { echo ""; echo "========================================"; echo "  $*"; echo "========================================"; }

for region in "${REGIONS[@]}"; do
    log "$region"

    echo "[1/3] Download..."
    python src/download.py --region "$region" --max-files "$MAX_FILES"

    echo "[2/3] DEM..."
    python src/download_dem.py

    echo "[3/3] Preprocess..."
    python src/preprocess.py --region "$region" --max-samples "$MAX_SAMPLES"

    echo "Removing raw .nc files..."
    rm -f "data/$region/"*.nc

    echo "Done: $region"
done

log "Combining regions..."
python src/preprocess.py --combine

log "Training..."
python src/train.py

log "Visualizing..."
python src/visualize.py

echo ""
echo "Pipeline complete."
