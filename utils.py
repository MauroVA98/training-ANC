import json
import os
from ast import literal_eval
from collections import namedtuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from finitediff import back_diff, central_diff
from filters import moving_average, butter_filter, OneEuroFilters

SubDataset = namedtuple("SubDataset", "X Y C meta")


def load_val(path: os.path, cfg, ratio: float = 1., show_progress: bool = True):
    """
    Return: List of Dictionaries with phi_data in ndarrays
    Inputs:
    path: Folder from where CSV data is loaded
    """
    path = os.path.join(os.getcwd(), path)
    filenames = []
    # find filenames that satisfy drone, trajectory, controller, wind conditions
    for dmg in os.listdir(path):
        for filename in os.listdir(os.path.join(path, dmg)):
            if filename.endswith(".csv"):
                filenames.append(os.path.join(dmg, filename))

    RawData = []
    filenames = sorted(filenames)
    for filename in tqdm(filenames, disable=not show_progress, desc=f"Loading test data from {path}"):
        # Load CSV data into Pandas Dataframe
        df = pd.read_csv(os.path.join(path, filename))
        RawData.append({})

        # Lists are loaded as strings by default, convert them back to lists
        for field in df.columns:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        # for idx in range(len(df)):
        #     df["p"][idx][2] = -df["p"][idx][2]
        #     df["v"][idx][2] = -df["v"][idx][2]
        #     df["p_d"][idx][2] = -df["p_d"][idx][2]
        #     df["v_d"][idx][2] = -df["v_d"][idx][2]

        # t = df["t"].values
        # v = np.stack(df["v"])
        # v_body = np.zeros(v.shape)
        # for i in range(len(df)):
        #     v_body[i, :] = np.array(df["R"].iloc[i]) @ v[i]
        #
        # df["v_body"] = v_body.tolist()

        calc_fr(df=df, cfg=cfg)
        # clean_LOS(filename=filename, df=df)

        thres = 1e-5
        start = np.argmax(np.linalg.norm(np.stack(df["v_d"]), axis=1) > thres)
        end = len(df["v_d"].values) - 1 - np.argmax(
            np.flip(np.linalg.norm(np.stack(df["v_d"]), axis=1)) > thres)

        df = df[start:end].reset_index(drop=True)
        df["t"] = df["t"] - df["t"][0]

        for field in df.columns:
            # Copy all the phi_data to a dictionary, and make things np.ndarrays
            RawData[-1][field] = np.array(df[field].tolist(), dtype=float)

        metadata_path = os.path.join(os.path.dirname(os.path.dirname(path)), "metadata", "0882412153.json")
        with open(metadata_path) as json_file:
            metadata = json.load(json_file)
            metadata["prop_dmg"] = filename.split("/")[0]
        RawData[-1]["metadata"] = metadata

    x = cfg["X"]
    y = cfg["Y"]
    Data = []
    for i, data in enumerate(RawData):
        # Create input array
        X = []
        for variable in x:
            if variable == "pwm" or variable == "rpm":
                X.append(data[variable] / 10000 * ratio)
            else:
                X.append(data[variable])
        X = np.hstack(X)

        # Create label array
        Y = data[y]

        # Pseudo-label for cross-entropy
        C = data["metadata"]["prop_dmg"]

        # Save to dataset
        Data.append(SubDataset(X, Y, C,
                               {"trajectory": RawData[i]["metadata"]["trajectory"]["name"],
                                "controller": RawData[i]["metadata"]["controller"],
                                "condition": RawData[i]["metadata"]["trajectory"]["vel_factor"], "t": RawData[i]["t"]}))
    return Data


def load_test(path: os.path, cfg, ratio: float = 1., drone=None, trajectory=None, controller=None, condition=None,
              show_progress: bool = True):
    """
    Return: List of Dictionaries with phi_data in ndarrays
    Inputs:
    folder: Folder from where CSV data is loaded
    drone: Load CSV data from specified Drone only
    trajectory: Load CSV data with specified Trajectory only
    controller: Load CSV data with specified Controller only
    condition: Load CSV data with specified Wind Conditions only
    """
    path = os.path.join(os.getcwd(), path)
    filenames = []
    # find filenames that satisfy drone, trajectory, controller, wind conditions
    for dmg in os.listdir(path):
        for filename in os.listdir(os.path.join(path, dmg)):
            if filename.endswith(".csv"):
                metadata_path = os.path.join(os.path.dirname(os.path.dirname(path)), "metadata",
                                             os.path.splitext("_".join([e for i, e in enumerate(filename.split("_"))
                                                                        if i in [0, 1, 2, 4, 5]]))[0] + ".json")
                with open(metadata_path) as json_file:
                    metadata = json.load(json_file)
                if check([metadata["drone"], metadata["trajectory"]["name"], metadata["controller"],
                          metadata["trajectory"]["vel_factor"]], [drone, trajectory, controller, condition]):
                    filenames.append(os.path.join(dmg, filename))

    RawData = []
    filenames = sorted(filenames)
    for filename in tqdm(filenames, disable=not show_progress, desc=f"Loading test data from {path}"):
        # Load CSV data into Pandas Dataframe
        df = pd.read_csv(os.path.join(path, filename))
        RawData.append({})

        # Lists are loaded as strings by default, convert them back to lists
        for field in df.columns:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        # for idx in range(len(df)):
        #     df["p"][idx][2] = -df["p"][idx][2]
        #     df["v"][idx][2] = -df["v"][idx][2]
        #     df["p_d"][idx][2] = -df["p_d"][idx][2]
        #     df["v_d"][idx][2] = -df["v_d"][idx][2]

        if "fr" not in df.columns:
            calc_fr(df=df, cfg=cfg)
        # clean_LOS(filename=filename, df=df)

        thres = 1e-5
        start = np.argmax(np.linalg.norm(np.stack(df["v_d"]), axis=1) > thres)
        end = len(df["v_d"].values) - 1 - np.argmax(
            np.flip(np.linalg.norm(np.stack(df["v_d"]), axis=1)) > thres)

        df = df[start:end].reset_index(drop=True)
        df["t"] = df["t"] - df["t"][0]

        for field in df.columns:
            # Copy all the phi_data to a dictionary, and make things np.ndarrays
            RawData[-1][field] = np.array(df[field].tolist(), dtype=float)

        metadata_path = os.path.join(os.path.dirname(os.path.dirname(path)), "metadata",
                                     os.path.splitext(
                                         "_".join([e for i, e in enumerate(filename.split("/")[1].split("_"))
                                                   if i in [0, 1, 2, 4, 5]]))[0] + ".json")
        with open(metadata_path) as json_file:
            metadata = json.load(json_file)
            metadata["prop_dmg"] = filename.split("/")[0]
        RawData[-1]["metadata"] = metadata

    x = cfg["X"]
    y = cfg["Y"]
    Data = []
    for i, data in enumerate(RawData):
        # Create input array
        X = []
        for variable in x:
            if variable == "pwm" or variable == "rpm":
                X.append(data[variable] / 10000 * ratio)
            else:
                X.append(data[variable])
        X = np.hstack(X)

        # Create label array
        Y = data[y]

        # Pseudo-label for cross-entropy
        C = data["metadata"]["prop_dmg"]

        # Save to dataset
        Data.append(SubDataset(X, Y, C,
                               {"trajectory": RawData[i]["metadata"]["trajectory"]["name"],
                                "controller": RawData[i]["metadata"]["controller"],
                                "condition": RawData[i]["metadata"]["trajectory"]["vel_factor"], "t": RawData[i]["t"]}))
    return Data


def load_train(path: os.path, cfg, drone=None, trajectory=None, controller=None, condition=None, show_progress=True):
    """
    Return: List of Dictionaries with phi_data in ndarrays
    Inputs:
    folder: Folder from where CSV data is loaded
    drone: Load CSV data from specified Drone only
    trajectory: Load CSV data with specified Trajectory only
    controller: Load CSV data with specified Controller only
    propeller damage: Load CSV data with specified propeller damage only
    speed factor: Load CSV data with specified speed factor only
    """

    path = os.path.join(os.getcwd(), path)
    filenames = []
    # find filenames that satisfy drone, trajectory, controller, wind conditions
    for dmg in os.listdir(path):
        for filename in os.listdir(os.path.join(path, dmg)):
            if filename.endswith(".csv"):
                metadata_path = os.path.join(os.path.dirname(os.path.dirname(path)), "metadata",
                                             os.path.splitext("_".join([e for i, e in enumerate(filename.split("_"))
                                                                        if i in [0, 1, 2, 4, 5]]))[0] + ".json")
                with open(metadata_path) as json_file:
                    metadata = json.load(json_file)
                if check([metadata["drone"], metadata["trajectory"]["name"], metadata["controller"],
                          metadata["trajectory"]["vel_factor"]], [drone, trajectory, controller, condition]):
                    filenames.append(os.path.join(dmg, filename))

    filenames = sorted(filenames)
    RawData = {k: [] for k in pd.read_csv(os.path.join(path, filenames[0]))}
    RawData["v_body"], RawData["v_filt"], RawData["fr"], RawData["prop_dmg"], RawData["Acc"], RawData["fr_filt"] = [], [], [], [], [], []
    for filename in tqdm(filenames, disable=not show_progress, desc=f"Loading train data from {path}"):
        # Load CSV data into Pandas Dataframe
        df = pd.read_csv(os.path.join(path, filename))
        metadata_path = os.path.join(os.path.dirname(os.path.dirname(path)), "metadata",
                                     os.path.splitext(
                                         "_".join([e for i, e in enumerate(filename.split("/")[1].split("_"))
                                                   if i in [0, 1, 2, 4, 5]]))[0] + ".json")
        with open(metadata_path) as json_file:
            metadata = json.load(json_file)

        # Lists are loaded as strings by default, convert them back to lists
        for idx, column in enumerate(df.columns):
            if isinstance(df[column][0], str):
                df[column] = df[column].apply(literal_eval)

        # for idx in range(len(df)):
        #     df["p"][idx][2] = -df["p"][idx][2]
        #     df["v"][idx][2] = -df["v"][idx][2]
        #     df["p_d"][idx][2] = -df["p_d"][idx][2]
        #     df["v_d"][idx][2] = -df["v_d"][idx][2]

        if "fr" not in df.columns:
            calc_fr(df=df, cfg=cfg)
        # clean_LOS(df=df)

        thres = 1e-5
        start = np.argmax(np.linalg.norm(np.stack(df["v_d"]), axis=1) > thres)
        end = len(df["v_d"].values) - 1 - np.argmax(np.flip(np.linalg.norm(np.stack(df["v_d"]), axis=1)) > thres)

        df = df[start:end].reset_index(drop=True)
        df["t"] = df["t"] - df["t"][0]

        for column in df.columns:
            RawData[column] += df[column].to_list()
        RawData["prop_dmg"] += [filename.split("/")[0] for i in range(len(df["t"]))]

    for key in RawData.keys():
        RawData[key] = np.array(RawData[key], dtype=float)
    cfg["train"] = {"trajectory": metadata["trajectory"]["name"], "controller": metadata["controller"]}
    return RawData


def format_train(RawData: dict["str", np.ndarray], cfg: dict, ratio: float = 1.) -> list[SubDataset]:
    x = cfg["X"]
    y = cfg["Y"]
    """
    Return: List of Dictionaries with SubDatasets Collated from RawData
    Inputs:
    RawData: List of Dictionaries with phi_data in ndarrays
    X: Variables to Collate to SubDatasets.X
    Y: Variable to Collate to SubDatasets.Y
    pwm_ratio: (avg. hover pwm of testing phi_data drone) / (avg. hover pwm of train phi_data drone)
    """

    # Create input array
    X = []
    for variable in x:
        if variable == "pwm" or variable == "rpm":
            X.append(RawData[variable] / 10000 * ratio)
        else:
            X.append(RawData[variable])
    X = np.hstack(X)

    # Create label array
    Y = RawData[y]

    Data = []
    prop_dmg = RawData["prop_dmg"]
    for j in range(cfg["nc"]):
        indices = np.where((j == prop_dmg))
        Data.append(SubDataset(X[indices], Y[indices], j, {"trajectory": cfg["train"]["trajectory"],
                                                           "controller": cfg["train"]["controller"],
                                                           "t": RawData["t"][indices]}))
    return Data


def calc_fr(df: pd.DataFrame, cfg: dict):
    m = 0.390       # 0.390 | 0.410
    g = 9.801

    k_t = 4.36e-08   # 4.36e-08 | 6.4e-08
    k_x = 1.08e-05
    k_y = 9.65e-06
    k_z = 2.79e-05
    k_h = 6.26e-02

    t = df["t"].values

    v = np.stack(df["v"])
    v_body = np.zeros(v.shape)

    for i in range(len(df)):
        v_body[i, :] = np.array(df["R"].iloc[i]) @ v[i]

    fa = np.zeros(v.shape)
    ft = np.zeros(v.shape)
    for i in range(len(df)):
        if cfg["amodel"]:
            fa_x = - k_x * v_body[i, 0] * np.sum(df["rpm"].iloc[i])
            fa_y = - k_y * v_body[i, 1] * np.sum(df["rpm"].iloc[i])
            fa_z = - k_z * v_body[i, 2] * np.sum(df["rpm"].iloc[i]) + k_h*(v_body[i, 0]**2 + v_body[i, 1]**2)
            fa[i, :] = [fa_x, fa_y, fa_z]
        ft[i, :] = [0, 0, -k_t * m * sum(i*i for i in df["rpm"].iloc[i])]

    if cfg["smoothing"]["velocity"] == "None":
        v_filt = v
    elif cfg["smoothing"]["velocity"]["type"] == "ma":
        v_filt = moving_average(v, n=cfg["smoothing"]["velocity"]["factor"], type="ma")
    elif cfg["smoothing"]["velocity"]["type"] == "ewma":
        v_filt = moving_average(v, n=cfg["smoothing"]["velocity"]["factor"], type="ewma")
    elif cfg["smoothing"]["velocity"]["type"] == "butter":
        v_filt = butter_filter(v, t=t, smoothing=cfg["smoothing"]["velocity"])
    elif cfg["smoothing"]["velocity"]["type"] == "1Euro":
        one_euro_vel_filter = OneEuroFilters(
            t[0], v[0],
            min_cutoff=cfg["smoothing"]["velocity"]["min_cutoff"],
            beta=cfg["smoothing"]["velocity"]["beta"],
            d_cutoff=cfg["smoothing"]["velocity"]["d_cutoff"])
        v_filt = one_euro_vel_filter.filter(t, v)

    if cfg["finite_diff"]["type"] == "backward":
        a_ned = back_diff(x=v_filt, t=t, order=cfg["finite_diff"]["order"], var=cfg["finite_diff"]["var"])
    if cfg["finite_diff"]["type"] == "central":
        a_ned = central_diff(x=v_filt, t=t, order=cfg["finite_diff"]["order"])


    fr = np.zeros((len(df), 3), dtype=float)
    for i in range(4, len(df)):
        fa_ned = np.array(df["R"].iloc[i]).T @ fa[i, :]
        ft_ned = np.array(df["R"].iloc[i]).T @ ft[i, :]
        fr[i] = m * (a_ned[i] - np.array([0, 0, g])) - ft_ned - fa_ned*m
    fr_filt = butter_filter(fr, t=t, smoothing=cfg["smoothing"]["acceleration"])

    df["v_filt"] = v_filt.tolist()
    df["v_body"] = v_body.tolist()
    df["fr"] = fr.tolist()
    df["fr_filt"] = fr_filt.tolist()
    df["Acc"] = a_ned.tolist()

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

    fig, ax = plt.subplots(nrows=1, figsize=(10, 6))

    ax.plot(df["t"], np.stack(df["v"])[:, 1], label="velocity")
    ax.plot(df["t"], np.stack(df["v_filt"])[:, 1], label="filtered velocity")
    ax.grid(True)
    ax.set_ylabel(r"$V \ [m/s]$")
    ax.set_xlabel(r"Time [s]")
    ax.legend(loc="upper right")
    ax.set_title("Velocity Measurement Filtering", fontsize=18)
    ax.set_xlim(left=70, right=100)
    ax.set_ylim(bottom=-1)

    axin = ax.inset_axes([0.18, 0.02, 0.2, 0.4])
    # Plot the data on the inset axis and zoom in on the important part
    axin.plot(df["t"], np.stack(df["v"])[:, 1], label="velocity")
    axin.plot(df["t"], np.stack(df["v_filt"])[:, 1], label="filtered velocity", color="#ff7f0e")
    x0, y0 = 83, -0.85
    width, height = 2.5, 0.3
    axin.set_xlim(x0, x0 + width)
    axin.set_ylim(y0, y0 + height)
    axin.set_xticks([])
    axin.set_yticks([])
    # Add the lines to indicate where the inset axis is coming from
    ax.indicate_inset_zoom(axin)

    plt.tight_layout()
    plt.show()


def check(metadata, args):
    cond = True
    for i in range(len(metadata)):
        if args[i] is None:
            continue
        elif isinstance(args[i], str) and args[i] == metadata[i]:
            continue
        elif isinstance(args[i], list) and any(x in metadata[i] for x in args[i]):
            continue
        else:
            cond = False
    return cond


def load_config(name):
    path = os.path.join(os.getcwd(), "configs", name)
    with open(path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def save_config(path, name, config):
    path = os.path.join(path, name + ".yaml")
    with open(path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def clean_LOS(df: pd.DataFrame):
    t = df["t"].values
    large_los = np.where(np.insert(t[1:] - t[:-1], 0, 0) > 0.2)[0]
    for idx in large_los:
        df.drop(df.loc[idx-5:idx+20].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # t = df["t"].values
    # small_los = np.where(((np.insert(t[1:] - t[:-1], 0, 0) > 0.05) | (np.insert(t[1:] - t[:-1], 0, 0) < 0.01)))[0]
    # diff = small_los[1:] - small_los[:-1]
    #
    # # Cleaning Optitrack Loss of Tracking
    # intervals = []
    # idx = 0
    # while idx < len(diff):
    #     i = 0
    #     while diff[idx + i] < 5:
    #         i += 1
    #         if idx + i == len(diff):
    #             break
    #     if i > 5:
    #         intervals.append([small_los[idx], small_los[idx + i]])
    #         idx += i
    #     else:
    #         idx += 1
    #
    # for interval in intervals:
    #     df.drop(df.loc[interval[0]:interval[1]].index, inplace=True)
    # df.reset_index(inplace=True, drop=True)


# if filename == manual.keys():
#     for interval in manual[filename]:
#         intervals.append(interval)

# formate_train()
# v_body = np.linalg.norm(RawData["v_body"], axis=1)
#
# length = len(v_body) / cfg["nc"]
# split = [np.min(v_body)]
# for i in range(cfg["nc"]):
#     width = 0.1
#     l = 0
#     while l < length:
#         if split[-1] + width > np.max(v_body):
#             break
#         l = len(np.where((split[-1] <= v_body) & (split[-1] + width > v_body))[0])
#         width += 0.005
#     split.append(split[-1] + width)
# cfg["split"] = split
#
# Data = []
# for j in range(cfg["nc"]):
#     indices = np.where((split[j] < v_body) & (v_body <= split[j + 1]))
#     Data.append(SubDataset(X[indices], Y[indices], j, {"trajectory": cfg["train"]["trajectory"],
#                                                        "controller": cfg["train"]["controller"],
#                                                        "condition": RawData["vel_factor"][indices],
#                                                        "t": RawData["t"][indices]}))
