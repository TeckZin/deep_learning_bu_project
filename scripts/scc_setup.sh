#!/bin/bash -l
# Environment setup and downloads only; this script does not train on the login node.
set -euo pipefail
config="${SCC_CONFIG:-$(cd "$(dirname "$0")" && pwd)/scc_config.local.sh}"
if [[ ! -f "$config" ]]; then
    echo "Copy scripts/scc_config.example.sh to scripts/scc_config.local.sh and edit it first." >&2
    exit 2
fi
source "$config"
if [[ -n "${SCC_PYTHON_MODULE:-}" ]]; then
    module load "$SCC_PYTHON_MODULE"
fi
"$SCC_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Configure SCC_PYTHON_MODULE/SCC_PYTHON for Python 3.12"'
mkdir -p "$(dirname "$SCC_ENV")" "$OUTPUT_ROOT" "$TORCH_HOME"
"$SCC_PYTHON" -m venv "$SCC_ENV"
source "$SCC_ENV/bin/activate"
cd "$SCC_REPO"
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements_midway.txt
python -m pip install --no-deps -e .
if [[ ! -f "$ENCODER_WEIGHTS" ]]; then
    python -m road_refinement prepare --encoder-weights "$ENCODER_WEIGHTS"
fi
python -m road_refinement check --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --encoder-weights "$ENCODER_WEIGHTS"
echo "Setup complete. Submit a smoke GPU job with: bash scripts/scc_submit.sh smoke"

