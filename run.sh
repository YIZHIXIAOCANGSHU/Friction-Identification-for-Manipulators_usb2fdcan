#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [[ ! -x ".venv/bin/mit-sender" ]]; then
    ./scripts/setup_venv.sh
fi

exec ./.venv/bin/mit-sender
