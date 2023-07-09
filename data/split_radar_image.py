import os
import numpy as np
import cv2
from tqdm import tqdm


def split_4_fold(image: np.ndarray):
    """
    Use to split the images to 4 400x400 size image
    to increase the processing speed
    """

    assert image.shape[0] == 1600
    assert image.shape[1] == 1600
    center = image.shape[0] // 2
    # split the image into 4 portion start from the top
    # centers_list = [(200, 800), (200, 1400), (1400, 200), (1400, 1400)])]
    center_list = [i for i in range(200, 1600, 400)]
    image_list = []
    for i in range(len(center_list)):
        croped_img = image[
            center_list[i] - 200 : center_list[i] + 200, center - 200 : center + 200
        ]
        image_list.append(croped_img)
    return image_list


radar_save_path = "/data/ITRI/scene_1/processed-1600/radar_400_crop"
radar_path = "/data/ITRI/scene_1/processed-1600/radar"
radar_paths = sorted(os.listdir(radar_path))

lidar_save_path = "/data/ITRI/scene_1/processed-1600/lidar_mask_400_crop"
lidar_mask_path = "/data/ITRI/scene_1/processed-1600/lidar_mask"
lidar_mask_paths = sorted(os.listdir(lidar_mask_path))

range_mask_base_path = "/data/ITRI/scene_1/processed-1600/"
range_mask_path = "/data/ITRI/scene_1/processed-1600/1600x1600_range_mask.png"

if not os.path.exists(radar_save_path):
    os.makedirs(radar_save_path)
if not os.path.exists(lidar_save_path):
    os.makedirs(lidar_save_path)


# for path in tqdm(radar_paths[3:]):
#     radar_filename = os.path.join(radar_path, path)

#     # radar_image = cv2.imread(radar_filename)
#     # radar_list = split_4_fold(radar_image)
#     # for i in range(len(radar_list)):
#     #     cv2.imwrite(
#     #         os.path.join(radar_save_path, path[:-4] + "_" + str(i) + ".png"),
#     #         radar_list[i],
#     #     )
#     lidar_mask_filename = os.path.join(lidar_mask_path, path)
#     lidar_mask_image = cv2.imread(lidar_mask_filename)

#     lidar_list = split_4_fold(lidar_mask_image)
#     for i in range(len(lidar_list)):
#         krenel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         lidar_morph = cv2.dilate(lidar_list[i], krenel, iterations=2)
#         cv2.imwrite(
#             os.path.join(lidar_save_path, path[:-4] + "_" + str(i) + ".png"),
#             lidar_morph,
# )
range_mask_image = cv2.imread(range_mask_path)
range_list = split_4_fold(range_mask_image)
for i in range(len(range_list)):
    cv2.imwrite(
        os.path.join(range_mask_base_path, "1600x1600_range_mask_" + str(i) + ".png"),
        range_list[i],
    )
    # for i in range(len(radar_list)):
    #     combine = cv2.addWeighted(radar_list[i], 0.8, lidar_morph, 0.8, 0)
    #     # cv2.imshow("combine", combine)
    #     # cv2.waitKey()

    # cv2.imshow("lidar", lidar_mask_image)
    # cv2.imshow("radar", radar_image)
    # cv2.waitKey()
