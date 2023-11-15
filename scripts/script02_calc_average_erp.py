import sys

sys.path.append("..")
import mne
from message_senders import LineSender
from rich.console import Console

from config import avg_erps_path, ica_epoch_path

console = Console()


def load_and_calc_avg(part_id: str):
    epochs = mne.read_epochs(ica_epoch_path(part_id))
    avg_move = epochs["probe_tone/move"].average()
    avg_no_move = epochs["probe_tone/no_move"].average()
    return avg_move, avg_no_move


def calc_avg_epochs(part_ids: "list[str]", musicians: bool):
    group = "m" if musicians else "nm"
    move_path = avg_erps_path(group, "move")
    no_move_path = avg_erps_path(group, "no_move")

    avg_move_list = []
    avg_no_move_list = []
    for part_id in part_ids:
        LineSender().send(part_id)
        try:
            avg_move, avg_no_move = load_and_calc_avg(part_id)
            avg_move_list.append(avg_move)
            avg_no_move_list.append(avg_no_move)
        except Exception:
            LineSender().send("exception at " + part_id)
            console.print_exception(show_locals=True)
            exit(code=1)

    avg_move = mne.combine_evoked(avg_move_list, weights="equal")
    avg_move.save(move_path, overwrite=True)
    avg_no_move = mne.combine_evoked(avg_no_move_list, weights="equal")
    avg_no_move.save(no_move_path, overwrite=True)
    return avg_move, avg_no_move


def main():
    LineSender().send("musicians starts")
    part_ids = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08"]
    calc_avg_epochs(part_ids=part_ids, musicians=True)

    LineSender().send("no musicians starts")
    part_ids = ["nm01", "nm02", "nm03", "nm04"]
    calc_avg_epochs(part_ids=part_ids, musicians=False)

    LineSender().send("終了しました")


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
