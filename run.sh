#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${1:-8000}"
BIND="${2:-127.0.0.1}"

if [ ! -d web/node_modules ]; then
  npm --prefix web install
fi

npm --prefix web run build
exec uv run main.py --port "$PORT" --bind "$BIND"
