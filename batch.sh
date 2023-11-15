#! /bin/bash
set -euxoC pipefail
cd "$(dirname "$0")"

err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2
}

### run python scripts
cd scripts
/opt/homebrew/Caskroom/mambaforge/base/envs/NIDlab/bin/python ./script01_load_ICA_epoching.py
/opt/homebrew/Caskroom/mambaforge/base/envs/NIDlab/bin/python ./script02_calc_average_erp.py
/opt/homebrew/Caskroom/mambaforge/base/envs/NIDlab/bin/python ./script03_erp_amp_late.py


