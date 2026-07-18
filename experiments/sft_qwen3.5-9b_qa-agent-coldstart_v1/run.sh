#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"

