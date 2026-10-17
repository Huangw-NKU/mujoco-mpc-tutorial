#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --locked mpc-demo run --scenario demo --viewer --out outputs/live "$@"
