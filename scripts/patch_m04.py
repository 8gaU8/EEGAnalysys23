# %%
# session1とsession2を連続して行ったため、1と2だけ別に処理する
import sys

sys.path.append("..")

import mne
import numpy as np
import pandas as pd

import eeg_utils
from config import ICA_EPOCHS_DIR

part_id = "nm04"


# %%
def parse_events(events: np.ndarray, df: pd.DataFrame):
    first_event = events[0]
    events_list = events

    # ブロックをトライアルに分割
    segment_list = []
    segment = []
    for start, end, event in events_list:
        if event == 5:
            segment = []
        segment.append([start, end, event])
        if event == 7:
            segment_list.append(segment)

    # 最大30試行に区切る
    segment_list = segment_list
    # Probe Toneをマーキング
    for i, segment in enumerate(segment_list):
        msg = 0
        if df.iloc[i]["msg"] == "MOVE":
            msg = 1
        elif df.iloc[i]["msg"] == "DON'T MOVE":
            msg = 2
        segment = list(map(lambda s: [s[0], s[1], s[2] + msg * 100], segment))

        segment_list[i] = segment

    # 条件ごとにイベントidを書き換え
    for i, segment in enumerate(segment_list):
        msg = 0
        if df.iloc[i]["msg"] == "MOVE":
            msg = 1
        elif df.iloc[i]["msg"] == "DON'T MOVE":
            msg = 2

        probe_id = 0
        fifth_seg_id = 0
        for j, event in enumerate(segment):
            if event[2] % 100 == 3:
                probe_id = j
            if event[2] % 100 == 1:
                fifth_seg_id = j

        segment[probe_id][2] = 11 + msg * 100
        segment[fifth_seg_id][2] = 12 + msg * 100
        segment_list[i] = segment

    events_list = [first_event]
    for segment in segment_list:
        events_list += segment
    events_list = np.array(events_list)
    return events_list


def main():
    if (ICA_EPOCHS_DIR / f"{part_id}-epo.fif.gz").exists():
        return
    # Session1を処理
    # %%
    df_exp1 = pd.read_csv("./exp_params/0.csv")
    df_exp2 = pd.read_csv("./exp_params/1.csv")
    event_df = pd.concat([df_exp1, df_exp2])

    # %%
    path = f"/Volumes/data/haga/data/{part_id}/eeg/session1.vhdr"
    original_raw_12 = mne.io.read_raw_brainvision(path, preload=True, verbose=False)

    # %%

    raw_12 = original_raw_12.copy()
    events, event_ids = mne.events_from_annotations(raw_12, verbose=False)
    events = parse_events(events, event_df)
    annot_from_events = eeg_utils.mk_annotated_events(events, raw_12)

    raw_12.set_annotations(annot_from_events)

    # %%
    # set event
    raw_12 = eeg_utils.apply_ref(raw_12)
    raw_12 = eeg_utils.set_montage(raw_12)

    # get_ica
    ica = eeg_utils.fit_ica(raw_12)
    raw_ica_12 = raw_12.copy()
    raw_ica_12.filter(l_freq=1, h_freq=None, n_jobs=10)
    raw_ica_12.notch_filter(freqs=60, notch_widths=0.5, n_jobs=10)
    ica.apply(raw_ica_12)
    raw_ica_12.set_annotations(annot_from_events)

    # %%
    epoch_12 = mne.Epochs(
        raw_ica_12,
        events,
        event_id=eeg_utils.event_dict,
        tmin=-0.1,
        tmax=0.6,
        verbose=False,
    )

    # %%
    #  session 3~5を 処理、1,2と分割
    exp_params_ids = [3, 4, 5]
    eeg_file_ids = ["3", "4", "5"]
    raw_ica_list = [raw_ica_12]
    epochs_list = [epoch_12]
    for exp_params_id, eeg_file_id in zip(exp_params_ids, eeg_file_ids):
        path = f"/Volumes/data/haga/data/{part_id}/eeg/session{eeg_file_id}.vhdr"
        print("loading", path)
        original_raw_12 = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        raw_12 = original_raw_12.copy()

        # perse event
        events, event_ids = mne.events_from_annotations(raw_12, verbose=False)
        event_df = pd.read_csv(f"./exp_params/{exp_params_id - 1}.csv")
        events = parse_events(events, event_df)
        annot_from_events = eeg_utils.mk_annotated_events(events, raw_12)
        # set event
        raw_12 = eeg_utils.apply_ref(raw_12)
        raw_12 = eeg_utils.set_montage(raw_12)

        # get_ica
        ica = eeg_utils.fit_ica(raw_12)
        raw_ica_12 = raw_12.copy()
        raw_ica_12.filter(l_freq=1, h_freq=None, verbose=False, n_jobs=10)
        raw_ica_12.notch_filter(freqs=60, notch_widths=0.5, verbose=False, n_jobs=10)
        ica.apply(raw_ica_12, verbose=False)
        raw_ica_12.set_annotations(annot_from_events)
        raw_ica_list.append(raw_ica_12)

        epoch_12 = mne.Epochs(
            raw_ica_12,
            events,
            event_id=eeg_utils.event_dict,
            tmin=-0.1,
            tmax=0.6,
            verbose=False,
        )
        epochs_list.append(epoch_12)

    # %%
    epochs = mne.concatenate_epochs(epochs_list)
    epochs.save(ICA_EPOCHS_DIR / f"{part_id}-epo.fif.gz", overwrite=False)


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
