import sys

import mne
import numpy as np
import pandas as pd
from message_senders import LineSender
from rich.console import Console

sys.path.append("..")
from config import const, event_dict
from eeg_utils import mk_annotated_events


def parse_events(event: np.ndarray, df: pd.DataFrame):
    MOVE_MSG = const.MOVE_MSG
    DONT_MOVE_MSG = const.DONT_MOVE_MSG
    DELAY_BIN_ORDER = const.DELAY_BIN_ORDER

    # Delayごとのbin id を
    delays = df["delay"].to_numpy()
    delay_bin = pd.cut(delays, const.NB_BIN)
    df.loc[:, ("delay_bin_id",)] = delay_bin
    df = df.replace(
        {"delay_bin_id": {d: i for i, d in enumerate(sorted(set(delay_bin)))}}
    )

    first_event = event[0]
    events_list = event

    # ブロックをトライアルに分割
    trial_list = []
    trial = []
    for start, end, event in events_list:
        if event == 5:
            trial = []
        trial.append([start, end, event])
        if event == 7:
            trial_list.append(trial)

    # 最大30試行に区切る
    trial_list = trial_list[-30:]

    # Probe Toneをマーキング
    for i, trial in enumerate(trial_list):
        msg = 0
        if df.iloc[i]["msg"] == "MOVE":
            msg = MOVE_MSG
        elif df.iloc[i]["msg"] == "DON'T MOVE":
            msg = DONT_MOVE_MSG
        trial = list(map(lambda s: [s[0], s[1], s[2] + msg], trial))

        trial_list[i] = trial

    # 条件ごとにイベントidを書き換え
    # Probe Toneと5小節目1拍目にイベント
    for i, trial in enumerate(trial_list):
        msg = 0
        if df.iloc[i]["msg"] == "MOVE":
            msg = MOVE_MSG
        elif df.iloc[i]["msg"] == "DON'T MOVE":
            msg = DONT_MOVE_MSG

        probe_id = 0
        fifth_seg_id = 0
        for event_id, event in enumerate(trial):
            if event[2] % 100 == 3:
                probe_id = event_id
            if event[2] % 100 == 1:
                fifth_seg_id = event_id

        trial[probe_id][2] = 11 + msg
        trial[fifth_seg_id][2] = 12 + msg
        trial_list[i] = trial

    for trial_id, trial in enumerate(trial_list):
        bin_id = df.loc[trial_id]["delay_bin_id"]
        for event_id, event in enumerate(trial):
            event[2] += bin_id * DELAY_BIN_ORDER
            trial_list[trial_id][event_id] = event

    events_list = [first_event]
    for trial in trial_list:
        events_list += trial
    events_array = np.array(events_list)
    return events_array


def valid_events(epoch, df):
    # df のほうからデータ取り出す
    delays = df["delay"].to_numpy()
    delay_bin = pd.cut(delays, const.NB_BIN)
    df.loc[:, ("delay_bin_id",)] = delay_bin
    df = df.replace(
        {"delay_bin_id": {d: i for i, d in enumerate(sorted(set(delay_bin)))}}
    )

    df_counts = df[df["msg"] == "MOVE"].groupby("delay_bin_id")
    counts_df = list(map(lambda s: len(s[1]), sorted(df_counts.indices.items())))

    # eventからデータ取り出す
    output_dict = dict(
        filter(lambda item: item[0].startswith("probe_tone/move"), event_dict.items())
    )
    output_dict = {v: k for k, v in output_dict.items()}

    array = epoch["probe_tone/move"].events
    result = mne.count_events(array, ids=output_dict)
    counts_epoch = list(result.values())
    assert counts_df == counts_epoch


def load_epoch(part_id, session_id):
    path = f"/Volumes/data/haga/data/{part_id}/eeg/session{session_id}.vhdr"
    original_raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)

    raw = original_raw.copy()
    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    df = pd.read_csv(f"../exp_params/{session_id-1}.csv")
    events = parse_events(events, df)
    annot_from_events = mk_annotated_events(events, raw)
    raw.set_annotations(annot_from_events)
    epoch = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=const.TMIN,
        tmax=const.TMAX,
        on_missing="ignore",
    )
    return epoch, df


def main():
    console = Console()
    part_ids = ["m01", "m02", "m03", "m04", "m05", "m06"]
    part_ids += ["nm01", "nm02", "nm03", "nm04"]
    for session_id in range(1, 6):
        for part_id in part_ids:
            try:
                epoch, df = load_epoch(part_id, session_id)
                valid_events(epoch, df)
            except Exception as e:
                LineSender().send("exception at " + part_id + ", " + str(e))
                print(e.__class__)


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
