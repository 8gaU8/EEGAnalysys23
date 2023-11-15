import sys
from functools import cache

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("..")

from config import erp_pkls_per_bin


def or_(picks):
    return "|".join(picks)


# @cache
def mean_amps(part_ids, conditions, bins, center, picks=None):
    amps = []
    for part_id in part_ids:
        df = pd.read_pickle(erp_pkls_per_bin(part_id, conditions, bins, center))
        if picks is not None:
            df = df[df["ch_names"].str.match(or_(picks))]
        amps.append(df["amplitudes"].mean(axis=0))
    return amps


def plot_vi(picks, center, conditions, name):
    part_m_ids = ["m01", "m02", "m03", "m04", "m05", "m06", "m08"]
    part_nm_ids = ["nm01", "nm02", "nm03", "nm04"]

    forward = ["0", "1", "2"]
    just = ["3", "4", "5"]
    delay = ["6", "7", "8"]

    cond_s = "+".join(conditions)
    for part_id, amp in zip(
        part_m_ids, mean_amps(part_m_ids, conditions, forward, center, picks)
    ):
        if amp > 0.01:
            print(part_id)
            print(part_m_ids, conditions, forward, center, picks)
    # fmt:off
    amps_m = {
        f"musician, {cond_s}, forward": mean_amps(part_m_ids, conditions, forward, center, picks),
        f"musician, {cond_s}, just": mean_amps(part_m_ids, conditions, just, center, picks),
        f"musician, {cond_s}, delay": mean_amps(part_m_ids, conditions, delay, center, picks),
    }

    df_m = pd.DataFrame(amps_m)
    df_m = df_m.melt()

    amps_nm = {
        f"non-musician, {cond_s}, forward": mean_amps(part_nm_ids, conditions, forward, center, picks),
        f"non-musician, {cond_s}, just": mean_amps(part_nm_ids, conditions, just, center, picks),
        f"non-musician, {cond_s}, delay": mean_amps(part_nm_ids, conditions, delay, center, picks),
    }
    df_nm = pd.DataFrame(amps_nm)
    df_nm = df_nm.melt()
    # fmt:on

    df = pd.concat([df_m, df_nm])
    df = df.rename(columns={"variable": "bins", "value": "amplitude"})

    cp = sns.color_palette("RdBu_r", 6)
    custom_colors = cp[:3] + list(reversed(cp[-3:]))

    sns.violinplot(
        x="bins",
        y="amplitude",
        data=df,
        palette=custom_colors,
        hue="bins",
    )
    plt.title(name)
    plt.ylim([-5 * 1e-6, 10 * 1e-6])
    plt.xticks(rotation=15)
    plt.savefig("../figs_per_bin/" + name.replace(" ", "_") + ".png")
    plt.close()


def main():
    # sns.set_style("whitegrid")
    plt.rcParams["figure.subplot.bottom"] = 0.2

    # fmt:off
    # picks_f = ["Fpz", "Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8"]
    # picks_r = ["FT8", "T8", "TP8", "FC6", "C6", "CP6"]
    # picks_l = ["FT7", "T7", "TP7", "FC5", "C5", "CP5"]
    # picks_b = ["Oz", "POz", "PO8", "PO3", "O1", "O9", "O2", "O10", "PO7"]
    picks_z= ["FCz", "Fz", "Cz"]
    # fmt:on

    pick_dict = {
        "all channels": None,
        # "frontal": picks_f,
        # "temporal": picks_r + picks_l,
        # "occiput": picks_b,
        "FCz, Fz, Cz": picks_z,
    }

    conditions_list = ["move"], ["no_move"], ["move", "no_move"]

    centers = 180 * 1e-3, 250 * 1e-3
    for conditions in conditions_list:
        for center in centers:
            for key in pick_dict.keys():
                picks = pick_dict[key]
                center_str = str(int(center * 1e3))
                title = key + ", " + center_str + ", (" + ",".join(conditions) + ")"
                plot_vi(picks, center, conditions, title)


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
