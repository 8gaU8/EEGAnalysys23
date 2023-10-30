from pathlib import Path
from typing import Literal

_INTER_FILES_ROOT = Path("/Volumes/data/haga/data/inter_files")

ICA_EPOCHS_DIR = _INTER_FILES_ROOT / "ica_epochs"
AVG_ERPS_DIR = _INTER_FILES_ROOT / "avg_erps"
ERP_CSVS_DIR = _INTER_FILES_ROOT / "erp_csvs"

_inter_dirs = [ICA_EPOCHS_DIR, AVG_ERPS_DIR, ERP_CSVS_DIR]
for dir in _inter_dirs:
    if not dir.exists():
        dir.mkdir(parents=True)


def ica_epoch(part_id: str) -> Path:
    path = ICA_EPOCHS_DIR / f"{part_id}-epo.fif.gz"
    return path


def evoked(
    group: Literal["m", "nm"],
    condition: Literal["move", "no_move"],
) -> Path:
    path = AVG_ERPS_DIR / f"avg_{condition}_{group}-ave.fif.gz"
    return path


def erp_pkls(
    part_id: str, condition: Literal["move", "no_move"], center: float
) -> Path:
    center = int(center * 1e3)
    path = ERP_CSVS_DIR / f"erp_{part_id}_{condition}_{center}-erp.pkl"
    return path
