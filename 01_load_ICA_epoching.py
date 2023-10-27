from message_senders import LineSender
from rich.console import Console

import eeg_utils
from config import ICA_EPOCHS


def save_and_notify(part_id, session_ids, file_ids):
    fname = Path(ICA_EPOCHS / f"{part_id}-epo.fif.gz")
    if fname.exists():
        LineSender().send(part_id + " exists")
        return
    epochs = eeg_utils.calc_epochs(part_id, session_ids, file_ids)
    epochs.save(ICA_EPOCHS / f"{part_id}-epo.fif.gz", overwrite=True)
    LineSender().send(part_id)


def main():
    console = Console()
    LineSender().send("開始します")

    params = [1, 2, 3, 4, 5]
    eegs = ["1", "2", "3", "4", "5"]
    default_params = {"params": params, "eegs": eegs}

    params_dict = {
        "m01": default_params,
        "m02": {"params": params, "eegs": ["1", "2", "3", "4", "5_re"]},
        "m03": default_params,
        "m04": default_params,
        "m05": default_params,
        "m06": default_params,
        "nm01": default_params,
        "nm02": default_params,
        "nm03": {"params": params, "eegs": ["1_rem", "2", "3", "4", "5"]},
        "nm04": default_params,
        "nm06": default_params,
    }
    for part_id in params_dict.keys():
        try:
            save_and_notify(
                part_id,
                params_dict[part_id]["params"],
                params_dict[part_id]["eegs"],
            )
        except Exception as e:
            LineSender().send("Exception at " + part_id)
            console.print_exception(show_locals=True)
            break

    LineSender().send("終了しました")


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
