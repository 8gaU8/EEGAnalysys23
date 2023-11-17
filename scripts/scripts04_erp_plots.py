import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("..")

from config import erp_pkls


def or_(picks):
    return "|".join(picks)


def mean_amps(part_ids, condition, center, picks=None):
    amps = []
    for part_id in part_ids:
        df = pd.read_pickle(erp_pkls(part_id, condition, center))
        if picks is not None:
            df = df[df["ch_names"].str.match(or_(picks))]
        amps.append(df["amplitudes"].mean(axis=0))
    return amps


def plot_vi(picks, center, name):
    part_m_ids = ["m01", "m02", "m03", "m04", "m05", "m06", "m07", "m08"]
    part_nm_ids = ["nm01", "nm02", "nm03", "nm04"]

    amps_m = {
        "musician, move": mean_amps(part_m_ids, "move", center, picks),
        "musican, don't move": mean_amps(part_m_ids, "no_move", center, picks),
    }
    df_m = pd.DataFrame(amps_m)
    df_m = df_m.melt()

    amps_nm = {
        "non-musician, move": mean_amps(part_nm_ids, "move", center, picks),
        "non-musician, don't move": mean_amps(part_nm_ids, "no_move", center, picks),
    }
    df_nm = pd.DataFrame(amps_nm)
    df_nm = df_nm.melt()

    df = pd.concat([df_m, df_nm])
    df = df.rename(columns={"variable": "condition", "value": "amplitude"})

    cp = sns.color_palette("RdBu_r", 24)
    custom_colors = cp[3], cp[9], cp[-4], cp[-9]

    plt.title(name)
    plt.ylim([-5 * 1e-6, 10 * 1e-6])
    sns.violinplot(
        x="condition",
        y="amplitude",
        data=df,
        palette=custom_colors,
        hue="condition",
    )
    plt.xticks(rotation=15)
    plt.savefig("../figs/" + name.replace(" ", "_") + ".png")
    plt.close()


def main():
    sns.set_style("whitegrid")
    plt.rcParams["figure.subplot.bottom"] = 0.2

    # fmt:off
    picks_f = ["Fpz", "Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8"]
    picks_r = ["FT8", "T8", "TP8", "FC6", "C6", "CP6"]
    picks_l = ["FT7", "T7", "TP7", "FC5", "C5", "CP5"]
    picks_b = ["Oz", "POz", "PO8", "PO3", "O1", "O9", "O2", "O10", "PO7"]
    # fmt:on

    pick_dict = {
        "all channels": None,
        "frontal": picks_f,
        "temporal": picks_r + picks_l,
        "occiput": picks_b,
    }
    centers = 180 * 1e-3, 250 * 1e-3, 300 * 1e-3
    for center in centers:
        for key in pick_dict.keys():
            picks = pick_dict[key]
            center_str = str(int(center * 1e3))
            title = key + ", " + center_str
            plot_vi(picks, center, title)


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).parent)
    main()
