import sys

sys.path.append("..")
from pathlib import Path
from typing import Literal

import mne
import pandas as pd
from message_senders import LineSender

from config import erp_pkls, ica_epoch_path


def get_each_erp(
    epochs: mne.EvokedArray, center: float, window_width: float
) -> pd.DataFrame:
    tmin = center - window_width
    tmax = center + window_width

    cropped = epochs.copy().crop(tmin, tmax)
    index_to_time = cropped.times
    indeces = cropped.get_data().argmax(axis=1)

    amplitudes = cropped.get_data().max(axis=1)
    latencies = index_to_time[indeces]
    ch_names = epochs.ch_names
    df = pd.DataFrame(
        zip(ch_names, amplitudes, latencies),
        columns=["ch_names", "amplitudes", "latencies"],
    )
    return df


def save_each_erp(
    epochs: mne.EvokedArray,
    part_id: str,
    condition: Literal["move", "no_move"],
    center: float,
    window_width: float = 20 * 1e-3,
) -> tuple[pd.DataFrame, Path]:
    avg_epochs = epochs[f"probe_tone/{condition}"].copy().average()
    df = get_each_erp(avg_epochs, center, window_width)
    path = erp_pkls(part_id, condition, center)
    df.to_pickle(path)
    return df, path


def main():
    # fmt:off
    part_ids = ["m01", "m02", "m03", "m04", "m05", "m06", "nm01", "nm02", "nm03", "nm04" ]
    # fmt:on
    LineSender().send("starts")
    for part_id in part_ids:
        epochs = mne.read_epochs(ica_epoch_path(part_id))

        center = 180 * 1e-3
        save_each_erp(epochs, part_id, "move", center)
        save_each_erp(epochs, part_id, "no_move", center)

        center = 250 * 1e-3
        save_each_erp(epochs, part_id, "move", center)
        save_each_erp(epochs, part_id, "no_move", center)

    LineSender().send("ends")


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
