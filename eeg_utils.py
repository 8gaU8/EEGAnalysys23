import mne
import mne_icalabel
import numpy as np
import pandas as pd
from mne import Annotations
from mne.io import RawArray
from mne.io.brainvision.brainvision import RawBrainVision
from mne.preprocessing import ICA

event_dict = {
    # "New Segment/": 99999,
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
    "move_tempo1": 101,
    "move_tempo234": 102,
    "move_tone": 103,
    "move_cross": 104,
    "move_start": 105,
    "move_stim_start": 106,
    "move_answer": 107,
    "move_ans_enabled": 109,
    "move_stim_end": 110,
    "move_probe_tone": 111,
    "move_5th_seg": 112,
    "no_move_tempo1": 201,
    "no_move_tempo234": 202,
    "no_move_tone": 203,
    "no_move_cross": 204,
    "no_move_start": 205,
    "no_move_stim_start": 206,
    "no_move_answer": 207,
    "no_move_ans_enabled": 209,
    "no_move_stim_end": 210,
    "no_move_probe_tone": 211,
    "no_move_5th_seg": 212,
}


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
    segment_list = segment_list[-30:]
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
        for s1, s2, s3 in segment:
            if s3 > 220:
                print(s3)

    events_list = [first_event]
    for segment in segment_list:
        events_list += segment
    events_list = np.array(events_list)
    return events_list


def mk_annotated_events(events: np.ndarray, raw: RawBrainVision) -> Annotations:
    mapping = {}
    for k in event_dict.keys():
        mapping[event_dict[k]] = k
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
        verbose=False,
    )
    return annot_from_events


def apply_ref(raw: RawBrainVision) -> RawArray:
    # apply ref
    ref_idx = raw.ch_names.index("X1")
    raw_ary = raw.get_data()
    raw_ary = raw_ary - raw_ary[ref_idx] // 2

    raw_array = RawArray(data=raw_ary, info=raw.info)
    return raw_array


def set_montage(raw: RawArray) -> RawArray:
    # set montage
    raw_copy = raw.copy()
    raw_copy.set_channel_types({"vEOG": "eog", "hEOG": "eog"})
    raw_copy.drop_channels("X1")
    montage = mne.channels.make_standard_montage("easycap-M1")
    raw_copy.set_montage(montage, verbose=False)
    return raw_copy


def fit_ica(raw: RawArray) -> ICA:
    ica = ICA(random_state=42, n_components=20)
    ica = mne.preprocessing.ICA(
        random_state=42,
        n_components=20,
        method="infomax",
        fit_params=dict(extended=True),
    )
    filt_raw = raw.copy()
    filt_raw.filter(1, 100, verbose=False, n_jobs=10)
    filt_raw.notch_filter(freqs=60, notch_widths=0.5, verbose=False, n_jobs=10)
    ica.fit(filt_raw, verbose=False)
    result = mne_icalabel.label_components(raw, ica, method="iclabel")
    labels = np.array(result["labels"])
    exclude = list(np.argwhere((labels != "other") & (labels != "brain")).flatten())
    ica.exclude = exclude
    return ica


def calc_epochs(part_id, exp_params_ids, eeg_file_ids):
    raw_ica_list = []
    epochs_list = []
    for exp_params_id, eeg_file_id in zip(exp_params_ids, eeg_file_ids):
        path = f"/Volumes/data/haga/data/{part_id}/eeg/session{eeg_file_id}.vhdr"
        print("loading", path)
        original_raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        raw = original_raw.copy()

        # perse event
        events, event_ids = mne.events_from_annotations(raw, verbose=False)
        event_df = pd.read_csv(f"./exp_params/{exp_params_id - 1}.csv")
        events = parse_events(events, event_df)
        annot_from_events = mk_annotated_events(events, raw)
        # set event
        raw = apply_ref(raw)
        raw = set_montage(raw)

        # get_ica
        ica = fit_ica(raw)
        raw_ica = raw.copy()
        raw_ica.filter(l_freq=1, h_freq=None, verbose=False, n_jobs=10)
        raw_ica.notch_filter(freqs=60, notch_widths=0.5, verbose=False, n_jobs=10)
        ica.apply(raw_ica, verbose=False)
        raw_ica.set_annotations(annot_from_events)
        raw_ica_list.append(raw_ica)

        epoch = mne.Epochs(
            raw_ica,
            events,
            event_id=event_dict,
            tmin=-0.1,
            tmax=0.6,
            verbose=False,
        )
        epochs_list.append(epoch)
    return mne.concatenate_epochs(epochs_list)
