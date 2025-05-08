#!/bin/bash
set -euo pipefail

pip install --no-deps -e /workspace/src/bnb
pip install --no-deps -e /workspace/src/bnb_intel

exec "$@"
