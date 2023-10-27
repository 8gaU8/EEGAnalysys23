import mne

from message_senders import LineSender

from config import AVG_ERPS, ICA_EPOCHS


def main():
    LineSender().send("musicians starts")
    part_ids = ("m01", "m02", "m03", "m04", "m05", "m06")
    avg_move_list = []
    avg_no_move_list = []
    for part_id in part_ids:
        LineSender().send(part_id)
        epochs = mne.read_epochs(ICA_EPOCHS / f"{part_id}-epo.fif.gz", verbose=False)
        avg_move = epochs["move_probe_tone"].average()
        avg_move_list.append(avg_move)
        avg_no_move = epochs["no_move_probe_tone"].average()
        avg_no_move_list.append(avg_no_move)

    avg_move_m = mne.combine_evoked(avg_move_list, weights="equal")
    avg_move_m.save(AVG_ERPS / "avg_move_m-epo.fif.gz")
    avg_no_move_m = mne.combine_evoked(avg_no_move_list, weights="equal")
    avg_no_move_m.save(AVG_ERPS / "avg_no_move_m-epo.fif.gz")

    LineSender().send("no musicians starts")
    part_ids = ("nm01", "nm02", "nm03", "nm04")
    avg_move_list = []
    avg_no_move_list = []
    for part_id in part_ids:
        LineSender().send(part_id)
        epochs = mne.read_epochs(ICA_EPOCHS / f"{part_id}-epo.fif.gz", verbose=False)
        avg_move = epochs["move_probe_tone"].average()
        avg_move_list.append(avg_move)
        avg_no_move = epochs["no_move_probe_tone"].average()
        avg_no_move_list.append(avg_no_move)

    avg_move_m = mne.combine_evoked(avg_move_list, weights="equal")
    avg_no_move_m.save(AVG_ERPS / "avg_no_move_nm-epo.fif.gz")
    avg_no_move_m = mne.combine_evoked(avg_no_move_list, weights="equal")
    avg_no_move_m.save(AVG_ERPS / "avg_no_move_nm-epo.fif.gz")


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
