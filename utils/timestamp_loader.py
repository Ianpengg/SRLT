import os
import numpy as np
from .registry import Registry

registry = Registry("timestamp_loader")


@registry.register()
def oxford_timestamp_loader(root_path):
    radar_folder = root_path
    timestamps_path = os.path.join(radar_folder, "radar.timestamps")
    radar_timestamps = np.loadtxt(
        timestamps_path, delimiter=" ", usecols=[0], dtype=np.int64
    )
    return radar_timestamps


@registry.register()
def radiate_timestamp_loader(root_path):
    radar_folder = root_path
    timestamps_path = os.path.join(radar_folder, "Navtech_Cartesian.txt")
    radar_timestamps = np.loadtxt(
        timestamps_path, delimiter=" ", usecols=[3], dtype=np.float64
    )
    return radar_timestamps


### Add your own timestamp loader here
# @registry.register()
# def your_timestamp_loader(root_path):
#    ...
