import sys
from pathlib import Path

import mne
import pandas as pd
from message_senders import LineSender

sys.path.append("..")
from config import erp_pkls_per_bin, ica_epoch_path
from eeg_utils import EpochReader, get_each_erp


def save_each_erp(
    epochs: mne.epochs.EpochsFIF,
    part_id: str,
    condition: list[str],
    bins: list[str],
    center: float,
    window_width: float = 20 * 1e-3,
) -> tuple[pd.DataFrame, Path]:
    avg_epochs = epochs["probe_tone"][condition][bins].copy().average()
    df = get_each_erp(avg_epochs, center, window_width)
    path = erp_pkls_per_bin(part_id, condition, bins, center)
    df.to_pickle(path)
    return df, path


def main():
    n = LineSender().send
    er = EpochReader()
    centers = (
        180 * 1e-3,
        300 * 1e-3,
        250 * 1e-3,
    )

    bin1 = ["0", "1", "2"]
    bin2 = ["3", "4", "5"]
    bin3 = ["6", "7", "8"]
    bins_list = bin1, bin2, bin3

    part_m_ids = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08"]
    part_nm_ids = ["nm01", "nm02", "nm03", "nm04"]
    part_ids = part_m_ids + part_nm_ids

    window_width = 20 * 1e-3

    conditions_list = ["move"], ["no_move"], ["move", "no_move"]
    n("start")
    for center in centers:
        for conditions in conditions_list:
            for part_id in part_ids:
                for bins in bins_list:
                    path = ica_epoch_path(part_id)
                    epochs = er.read(path)
                    save_each_erp(
                        epochs, part_id, conditions, bins, center, window_width
                    )
    n("end")


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
