import mne
import mne_icalabel
import numpy as np
import pandas as pd
from mne import Annotations
from mne.io import RawArray
from mne.io.brainvision.brainvision import RawBrainVision
from mne.preprocessing import ICA

from config import const, event_dict


class Singleton:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            # Singletonクラスはobjectクラスを継承しているため、super()はobjectを指す
            cls.instance = super().__new__(cls)
            print("インスタンスを作成")
        return cls.instance


class EpochReader(Singleton):
    def __init__(self) -> None:
        self.loaded = {}

    def read(self, path) -> mne.epochs.EpochsFIF:
        if path in self.loaded.keys():
            print("using cached data")
            return self.loaded[path]
        epoch = mne.read_epochs(path)
        self.loaded[path] = epoch
        return epoch

    def clear(self):
        self.loaded = {}


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

    # セッションをトライアルに分割
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

    # 条件毎にトリガーの値を変更
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

    # delay bin 番号をイベントに割り振る
    for trial_id, trial in enumerate(trial_list):
        bin_id = df.iloc[trial_id]["delay_bin_id"]
        for event_id, event in enumerate(trial):
            event[2] += bin_id * DELAY_BIN_ORDER
            trial_list[trial_id][event_id] = event

    events_list = [first_event]
    for trial in trial_list:
        events_list += trial
    events_array = np.array(events_list)
    return events_array


def mk_annotated_events(events: np.ndarray, raw: RawBrainVision) -> Annotations:
    mapping = {}
    for k in event_dict.keys():
        mapping[event_dict[k]] = k
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )
    return annot_from_events


def apply_ref(raw: RawBrainVision) -> RawArray:
    # apply ref
    ref_idx = raw.ch_names.index("X1")
    raw_ary = raw.get_data()
    raw_ary = raw_ary - raw_ary[ref_idx] / 2

    raw_array = RawArray(data=raw_ary, info=raw.info)
    return raw_array


def set_montage(raw: RawArray) -> RawArray:
    # set montage
    raw_copy = raw.copy()
    raw_copy.set_channel_types({"vEOG": "eog", "hEOG": "eog"})
    raw_copy.drop_channels("X1")
    montage = mne.channels.make_standard_montage("easycap-M1")
    raw_copy.set_montage(montage)
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
    filt_raw.filter(1, 100, n_jobs=10)
    filt_raw.notch_filter(freqs=60, notch_widths=0.5, n_jobs=10)
    ica.fit(filt_raw)
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
        original_raw = mne.io.read_raw_brainvision(path, preload=True)
        raw = original_raw.copy()

        # perse event
        events, event_ids = mne.events_from_annotations(raw)
        event_df = pd.read_csv(f"../exp_params/{exp_params_id - 1}.csv")
        events = parse_events(events, event_df)
        annot_from_events = mk_annotated_events(events, raw)
        # set event
        raw = apply_ref(raw)
        raw = set_montage(raw)

        # get_ica
        ica = fit_ica(raw)
        raw_ica = raw.copy()
        raw_ica.filter(l_freq=1, h_freq=None, n_jobs=10)
        raw_ica.notch_filter(freqs=60, notch_widths=0.5, n_jobs=10)
        ica.apply(raw_ica)
        raw_ica.set_annotations(annot_from_events)
        raw_ica_list.append(raw_ica)

        epoch = mne.Epochs(
            raw_ica,
            events,
            event_id=event_dict,
            tmin=-0.1,
            tmax=0.6,
            on_missing="warn",
            baseline=None,
        )
        epochs_list.append(epoch)
    return mne.concatenate_epochs(epochs_list)
