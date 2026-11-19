#!/usr/bin/env bash
# Create an isolated working copy for one arm of the experiment.
#
#   society/new_arm.sh single  ../arm-single
#   society/new_arm.sh society ../arm-society
#
# Each arm gets its own clone, its own `champion` branch starting from the same
# commit, and its own tracking files, so the two arms cannot see each other.
set -euo pipefail
ARM="$1"; DEST="$2"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
START="${START_COMMIT:-$(git -C "$SRC" rev-parse HEAD)}"

git clone -q "$SRC" "$DEST"
cd "$DEST"
git checkout -q "$START"
git branch -f champion "$START"
ln -s "$SRC/.venv" .venv 2>/dev/null || true
[ -f "$SRC/.mcp.json" ] && cp "$SRC/.mcp.json" . || true

if [ "$ARM" = "single" ]; then
  cat > arm.env <<EOT
ARM=single
LABS="solo"
THEMES="any"
PER_CYCLE=4
EOT
else
  cat > arm.env <<EOT
ARM=society
LABS="attn mlp resid embed"
THEMES="attention and positional encoding;MLP and activation functions;residual stream and normalisation;embeddings, output head and training-loop logic"
PER_CYCLE=4
EOT
fi
echo "Created $ARM arm at $DEST from $(git rev-parse --short "$START")"
