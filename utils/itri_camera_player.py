import cv2
import os
import numpy as np
from datetime import datetime
from icecream import ic


def onTrackbarChange(idx):
    camera_idx, camera_timestamp = get_sync(
        radar_timestamps[idx] + dif, camera_timestamps
    )

    camera_filename = os.path.join(image_path, f"{camera_timestamp}.jpeg")
    camera_image = cv2.imread(camera_filename, 1)
    camera_image = cv2.resize(camera_image, (512, 512))

    radar_filename = os.path.join(det_motion_path, f"{radar_timestamps[idx]}.png")
    radar_image = cv2.imread(radar_filename, 1)

    combine = np.concatenate((camera_image, radar_image), axis=1)
    cv2.imshow("combine", combine)


def get_sync(target_ts, total_ts):
    """
    get the closest id in timestamps given the target_ts
    :param t: timestamp in seconds
    :
    :return: the closest id
    :rtype: int
    """

    idx = np.argmin(np.abs(total_ts - target_ts))
    return idx, total_ts[idx]


def UnixTimeToSec(unix_timestamp):
    time = datetime.fromtimestamp(unix_timestamp / 1000000000)
    s = unix_timestamp % 1000000000
    sec_timestamp = (
        time.hour * 3600 + time.minute * 60 + time.second + (float(s) / 1000000000)
    )
    # sec_timestamp = (time - datetime.datetime(1970, 1, 1)).total_seconds() + (float(s) / 1000000)
    return sec_timestamp


data_path = "/data/ITRI"
seq = "scene"
debug = True
det_motion_path = (
    "/data/Codespace/RaMNet/logs/compare_output/fintune-500-epoch02-0649-mask"
)
# mask_only_result_path = "/data/Codespace/RaMNet/logs/output_result/oxford-epoch18-0583"
# det_motion_path = os.path.join(det_motion_path, seq)
# det_motion_path = mask_only_result_path
radar_path = os.path.join(data_path, seq)
radar_path = os.path.join(radar_path, "processed-512/radar")


image_path = os.path.join(data_path, seq)
image_path = os.path.join(image_path, "gige_3")

camera_timestamps_path = os.path.join(data_path + f"/{seq}", "gige_3.timestamps")
radar_timestamps_path = os.path.join(data_path + f"/{seq}", "radar.timestamps")

camera_timestamps = np.loadtxt(camera_timestamps_path, dtype=np.int64)
radar_timestamps = np.loadtxt(radar_timestamps_path, dtype=np.int64)


dif = (
    camera_timestamps[40] - radar_timestamps[0]
)  # compensate the time gap between camera and radar timestamps

target = sorted(os.listdir(det_motion_path))[0].split(".")[0]
idx, radar_timestamp = get_sync(np.int64(target), radar_timestamps)
radar_timestamps = radar_timestamps[idx:]
idx = 5
cv2.namedWindow("combine", cv2.WINDOW_NORMAL)
cv2.resizeWindow("combine", 2048, 1024)
cv2.createTrackbar("Slider", "combine", idx, len(radar_timestamps), onTrackbarChange)

while True:
    camera_idx, camera_timestamp = get_sync(
        radar_timestamps[idx] + dif, camera_timestamps
    )
    camera_filename = os.path.join(image_path, f"{camera_timestamp}.jpeg")
    camera_image = cv2.imread(camera_filename, 1)
    camera_image = cv2.resize(camera_image, (512, 512))

    radar_filename = os.path.join(det_motion_path, f"{radar_timestamps[idx]}.jpg")
    radar_image = cv2.imread(radar_filename, 1)

    if debug:
        ic(UnixTimeToSec(radar_timestamps[idx] + dif))
        ic(UnixTimeToSec(camera_timestamp))
        ic(radar_filename)
        ic(UnixTimeToSec(radar_timestamps[idx]), UnixTimeToSec(camera_timestamp))

    combine = np.concatenate((camera_image, radar_image), axis=1)
    cv2.imshow("combine", combine)
    cv2.waitKey(10)
    cv2.setTrackbarPos("Slider", "combine", idx)

    key = cv2.waitKey()
    if key == 100:
        idx += 1
        idx = min(idx, len(camera_timestamps))
    elif key == 97:
        idx -= 1
        idx = max(idx, 4)
