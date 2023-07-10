import os
import numpy as np
import cv2

# combine the split mask into one mask
def combine_split_mask(start_idx, end_idx, radar_timestamps):
    mask_range = [(0, 400), (400, 800), (800, 1200), (1200, 1600)]
    for timestamp in radar_timestamps[start_idx:end_idx]:
        mask_combined = np.zeros((1600, 1600), dtype=np.uint8)
        for i in range(0, 4):
            mask_filename = os.path.join(mask_folder, str(timestamp)+  f"_{i}.png")
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            mask_combined[mask_range[i][0]:mask_range[i][1], 800-200: 800+200] = mask

        cv2.imshow("mask", mask_combined)
        cv2.waitKey(0)
        
        cv2.imwrite(os.path.join(mask_folder, str(timestamp)+".png"), mask_combined)






mask_folder = "/data/ITRI/scene_1/processed-1600/mask/"
radar_timestamps = np.loadtxt("/data/ITRI/scene_1/radar.timestamps", dtype=np.int64)

combine_split_mask(2400, 2460, radar_timestamps)
