#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

export ROBOTWIN_OPENCLAW_MIRROR_DIR="${ROBOTWIN_OPENCLAW_MIRROR_DIR:-$ROOT_DIR/results/openclaw_bridge}"
export ROBOTWIN_OPENCLAW_MIRROR_EVERY="${ROBOTWIN_OPENCLAW_MIRROR_EVERY:-1}"

cd "$SCRIPT_DIR"
exec bash eval.sh "$@"
