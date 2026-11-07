#!/usr/bin/env bash
# Builds a fake home folder full of typical clutter, so you can record the demo GIF
# (and test `clean`) without touching or showing your real files. Uses about 1.9 GB.
#
#   bash demo/make_demo_home.sh           # creates /tmp/disksage-demo
#   HOME=/tmp/disksage-demo disksage      # scan it
#   rm -rf /tmp/disksage-demo             # clean up afterwards
set -euo pipefail

DEMO="${1:-/tmp/disksage-demo}"
case "$DEMO" in
  /tmp/?*|/private/tmp/?*) ;;
  *) echo "Refusing to use '$DEMO': pick a folder under /tmp." >&2; exit 1 ;;
esac
rm -rf "$DEMO"

# blob PATH MB: random bytes, so nothing compresses and sizes look real
blob() { mkdir -p "$(dirname "$1")"; head -c "$(( $2 * 1024 * 1024 ))" /dev/urandom > "$1"; }
# stale PATH...: mark files as last touched in January 2026
stale() { touch -t 202601150900 "$@"; }
sha256() { (shasum -a 256 "$1" 2>/dev/null || sha256sum "$1") | cut -d' ' -f1; }

# Caches that rebuild themselves
blob "$DEMO/Library/Caches/Homebrew/downloads/node--24.tar.gz" 120
blob "$DEMO/Library/Caches/pip/http/wheels.bin" 90
blob "$DEMO/Library/Developer/Xcode/DerivedData/MyApp-abcdef/Build/Intermediates.noindex/main.o" 220

# Cruft in projects nobody has opened for months
mkdir -p "$DEMO/Projects/old-react-app" "$DEMO/Projects/ml-experiments"
printf '{"name": "old-react-app"}\n' > "$DEMO/Projects/old-react-app/package.json"
blob "$DEMO/Projects/old-react-app/node_modules/.cache/bundle.bin" 140
printf '[project]\nname = "ml-experiments"\n' > "$DEMO/Projects/ml-experiments/pyproject.toml"
blob "$DEMO/Projects/ml-experiments/.venv/lib/python3.12/site-packages/torch/lib/libtorch.dylib" 260
stale "$DEMO/Projects/old-react-app/package.json" "$DEMO/Projects/ml-experiments/pyproject.toml"

# The same model, downloaded by three apps. `cat` makes real copies, not clones.
blob "$DEMO/model.tmp" 300
digest=$(sha256 "$DEMO/model.tmp")
size=$(wc -c < "$DEMO/model.tmp" | tr -d ' ')

ollama="$DEMO/.ollama/models"
mkdir -p "$ollama/blobs" "$ollama/manifests/registry.ollama.ai/library/llama3.2"
cat "$DEMO/model.tmp" > "$ollama/blobs/sha256-$digest"
printf '{"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:%s","size":%s}]}\n' \
  "$digest" "$size" > "$ollama/manifests/registry.ollama.ai/library/llama3.2/latest"

hf="$DEMO/.cache/huggingface/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF"
mkdir -p "$hf/blobs" "$hf/snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8"
cat "$DEMO/model.tmp" > "$hf/blobs/$digest"
ln -s "../../blobs/$digest" "$hf/snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

lms="$DEMO/.lmstudio/models/bartowski/Llama-3.2-3B-Instruct-GGUF"
mkdir -p "$lms"
cat "$DEMO/model.tmp" > "$lms/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
rm "$DEMO/model.tmp"

# A big file that's yours: disksage must call it REVIEW, never SAFE
blob "$DEMO/Documents/datasets/survey-2025.parquet" 150

echo "Demo home ready at $DEMO ($(du -sh "$DEMO" | cut -f1))."
echo "Scan it with:  HOME=$DEMO disksage"
