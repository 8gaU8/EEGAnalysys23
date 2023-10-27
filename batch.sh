#! /bin/bash
set -euxoC pipefail
cd "$(dirname "$0")"

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

/opt/homebrew/Caskroom/mambaforge/base/envs/NIDlab/bin/python 01_load_ICA_epoching.py
/opt/homebrew/Caskroom/mambaforge/base/envs/NIDlab/bin/python 02_calc_average_erp.py


