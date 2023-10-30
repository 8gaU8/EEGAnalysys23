import sys

sys.path.append("..")

from message_senders import LineSender
from rich.console import Console

import eeg_utils
from config import ICA_EPOCHS_DIR


def save_and_notify(part_id, params, eegs, patch_func):
    fname = ICA_EPOCHS_DIR / f"{part_id}-epo.fif.gz"
    if fname.exists():
        LineSender().send(part_id + " exists")
        return

    if patch_func is not None:
        patch_func()
        return

    epochs = eeg_utils.calc_epochs(part_id, params, eegs)
    epochs.save(fname)
    LineSender().send(part_id)


def get_m02_patch():
    from patch_m04 import main

    return main


def main():
    console = Console()
    LineSender().send("開始します")

    params = [1, 2, 3, 4, 5]
    eegs = ["1", "2", "3", "4", "5"]
    default_params = {"params": params, "eegs": eegs, "patch_func": None}

    # fmt: off
    params_dict = {
        "m01": default_params,
        "m02": {"params": params, "eegs": ["1", "2", "3", "4", "5_re"], "patch_func": None},
        "m03": default_params,
        "m04": default_params,
        "m05": default_params,
        "m06": default_params,
        "nm01": default_params,
        "nm02": default_params,
        "nm03": {"params": params, "eegs": ["1_rem", "2", "3", "4", "5"], "patch_func": None},
        "nm04": {"params": params, "eegs": eegs, "patch_func": get_m02_patch()},
        "nm06": default_params,
    }
    # fmt： on

    for part_id in params_dict.keys():
        try:
            save_and_notify(part_id=part_id, **params_dict[part_id])
        except Exception as e:
            LineSender().send("Exception at " + part_id)
            console.print_exception(show_locals=True)
            exit(code=1)

    LineSender().send("終了しました")


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
