from pathlib import Path
from typing import Literal, NamedTuple


#######* CONSTS
class _consts(NamedTuple):
    NB_BIN = 7
    MOVE_MSG = 100
    DONT_MOVE_MSG = 200
    DELAY_BIN_ORDER = 1000


const = _consts()

#######* event_dict
_base_event_dict = {
    "tempo1": 1,
    "tempo234": 2,
    "tone": 3,
    "cross": 4,
    "start": 5,
    "stim_start": 6,
    "answer": 7,
    "ans_enabled": 9,
    "stim_end": 10,
    "probe_tone": 11,
    "5th_seg": 12,
}

event_dict = {}
for k, v in _base_event_dict.items():
    for c in "rest", "move", "no_move":
        kc = k + "/" + c
        kv = v
        if c == "move":
            kv += const.MOVE_MSG
        if c == "no_move":
            kv += const.DONT_MOVE_MSG

        for bin_id in range(const.NB_BIN):
            kcb = kc + "/" + str(bin_id)
            vcb = kv + bin_id * const.DELAY_BIN_ORDER
            event_dict[kcb] = vcb


#######* path

_INTER_FILES_ROOT = Path("/Volumes/data/haga/data/inter_files_1109")


ICA_EPOCHS_DIR = _INTER_FILES_ROOT / "ica_epochs"
AVG_ERPS_DIR = _INTER_FILES_ROOT / "avg_erps"
ERP_CSVS_DIR = _INTER_FILES_ROOT / "erp_csvs"
FIG_DIR = Path("../figs")

_inter_dirs = [ICA_EPOCHS_DIR, AVG_ERPS_DIR, ERP_CSVS_DIR, FIG_DIR]
for dir in _inter_dirs:
    if not dir.exists():
        dir.mkdir(parents=True)


def ica_epoch_path(part_id: str) -> Path:
    path = ICA_EPOCHS_DIR / f"{part_id}-epo.fif.gz"
    return path


def avg_erps_path(
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
