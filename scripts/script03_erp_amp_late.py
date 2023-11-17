import sys
from tqdm import tqdm
from pathlib import Path
from typing import Literal

import mne
import pandas as pd
from message_senders import LineSender

sys.path.append("..")

from config import erp_pkls, ica_epoch_path
from eeg_utils import get_max_erp, get_min_erp


def save_each_erp(
    epochs: mne.EvokedArray,
    part_id: str,
    condition: Literal["move", "no_move"],
    center: float,
    window_width: float = 20 * 1e-3,
    min=False,
) -> tuple[pd.DataFrame, Path]:
    avg_epochs = epochs[f"probe_tone/{condition}"].copy().average()
    if not min:
        df = get_max_erp(avg_epochs, center, window_width)
    else:
        df = get_min_erp(avg_epochs, center, window_width)
    path = erp_pkls(part_id, condition, center)
    df.to_pickle(path)
    return df, path


def main():
    part_m_ids = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08"]
    part_nm_ids = ["nm01", "nm02", "nm03", "nm04"]
    part_ids = part_m_ids + part_nm_ids

    LineSender().send("starts")
    for part_id in tqdm(part_ids):
        epochs = mne.read_epochs(ica_epoch_path(part_id))

        center = 180 * 1e-3
        save_each_erp(epochs, part_id, "move", center, min=True)
        save_each_erp(epochs, part_id, "no_move", center, min=True)

        center = 250 * 1e-3
        save_each_erp(epochs, part_id, "move", center)
        save_each_erp(epochs, part_id, "no_move", center)

        center = 300 * 1e-3
        save_each_erp(epochs, part_id, "move", center)
        save_each_erp(epochs, part_id, "no_move", center)

    LineSender().send("ends")


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
