#!/bin/bash
# Copy a small, deterministic slice of the RoadSeg dataset from SCC for local dev.
# Takes the first N tiles (sorted by filename) per domain, with matching masks.
#
# Usage: scripts/fetch_data_slice.sh [N=25] [SPLIT=Val] [HOST=scc]
#   N      tiles per domain (Urban and Rural each)
#   SPLIT  Train or Val
#   HOST   ssh host alias (see README "SCC access")
# Output: project_a/data/<SPLIT>/{Urban,Rural}/{images_png,masks_png}/ (gitignored)
set -euo pipefail

N="${1:-25}"
SPLIT="${2:-Val}"
HOST="${3:-scc}"
REMOTE=/projectnb/ec523kb/projects/RoadSeg/dataset
DEST="$(cd "$(dirname "$0")/.." && pwd)/project_a/data"

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

ssh "$HOST" "cd $REMOTE && for a in Urban Rural; do
  ls $SPLIT/\$a/images_png | sort | head -n $N | while read f; do
    echo $SPLIT/\$a/images_png/\$f; echo $SPLIT/\$a/masks_png/\$f
  done
done" > "$LIST"

mkdir -p "$DEST"
rsync -a --files-from="$LIST" "$HOST:$REMOTE/" "$DEST/"

echo "Copied $(wc -l < "$LIST" | tr -d ' ') files ($N per domain, $SPLIT) to $DEST"
du -sh "$DEST"
