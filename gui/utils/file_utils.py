import numpy as np
import os
import cv2


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def file_to_id(str_):
    idx = str_[find(str_, "/")[-1] + 1 : str_.find(".npy")]
    idx = int(idx)
    return idx


class DataLoader:
    def __init__(self, file_path, radar_timestamps):
        self.file_path = file_path
        self.timestamp = radar_timestamps
        self.id = None
        self.data = None
        self.image = None
        self.mask = None
        self.mask_path = None

    def is_valid(self):
        return os.path.exists(self.image_path)

    def load_data(self, new_id):
        self.id = new_id
        str_format = "{:06d}"
        self.image_path = (
            self.file_path + "radar/" + str(self.timestamp[self.id]) + ".png"
        )

        if os.path.exists(self.image_path):
            pass
        else:
            self.image_path = (
                self.file_path + "radar/" + str_format.format(self.id) + ".png"
            )

        self.mask_path = (
            self.file_path + "mask/" + str(self.timestamp[self.id]) + ".png"
        )
        mask_folder = os.path.dirname(self.mask_path)
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)

        self.lidar_mask_path = (
            self.file_path + "lidar_mask/" + str(self.timestamp[self.id]) + ".png"
        )
        self.range_mask_path = self.file_path + "1600x1600_range_mask.png"

    def load_image(self):
        if self.is_valid():
            self.image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            return self.image
        else:
            return None

    def load_mask(self):
        if not os.path.exists(self.mask_path):
            self.mask = np.zeros(self.image.shape[:2])
            return self.mask
        else:
            self.mask = cv2.imread(self.mask_path, 0)
            self.mask = (self.mask / 255).astype(np.float32)

            return self.mask

    def load_lidar_mask(self):
        if not os.path.exists(self.lidar_mask_path):
            self.lidar_mask = np.zeros(self.image.shape[:2])

            return self.lidar_mask
        else:
            self.lidar_mask = cv2.imread(self.lidar_mask_path, 1)
            self.lidar_mask = cv2.cvtColor(self.lidar_mask, cv2.COLOR_BGR2RGB)
            self.range_mask = cv2.imread(self.range_mask_path, 1)
            self.range_mask = cv2.cvtColor(self.range_mask, cv2.COLOR_BGR2RGB)
            combine = cv2.addWeighted(self.lidar_mask, 1, self.range_mask, 0.8, 0)
            return combine

    def save_mask(self, mask):
        cv2.imwrite(self.mask_path, 255 * mask.astype(np.uint8))


class Patch_DataLoader:
    def __init__(self, file_path, radar_timestamps, patch_num):
        self.file_path = file_path
        self.timestamp = radar_timestamps
        self.id = None
        self.data = None
        self.image = None
        self.mask = None
        self.mask_path = None
        self.patch_num = patch_num

    def is_valid(self):
        return os.path.exists(self.image_path)

    def load_data(self, new_id):
        self.id = new_id

        # Add the patch image lodaing method, which is named with format like "timestamp_{patch_num}.png"
        self.image_path = (
            self.file_path
            + "radar_400_crop/"
            + str(self.timestamp[self.id])
            + f"_{self.patch_num}"
            + ".png"
        )

        # Add the patch mask loading method, which is named with format like "timestamp_{patch_num}.png"
        self.mask_path = (
            self.file_path
            + "mask/"
            + str(self.timestamp[self.id])
            + f"_{self.patch_num}"
            + ".png"
        )
        mask_folder = os.path.dirname(self.mask_path)
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)

        self.lidar_mask_path = (
            self.file_path
            + "lidar_mask_400_crop/"
            + str(self.timestamp[self.id])
            + f"_{self.patch_num}"
            + ".png"
        )
        self.lidar_range_mask_path = (
            self.file_path + f"1600x1600_range_mask_{self.patch_num}.png"
        )

    def load_image(self):
        if self.is_valid():
            self.image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            return self.image
        else:
            return None

    def load_mask(self):
        if not os.path.exists(self.mask_path):
            self.mask = np.zeros(self.image.shape[:2])
            return self.mask
        else:
            self.mask = cv2.imread(self.mask_path, 0)
            self.mask = (self.mask / 255).astype(np.float32)

            return self.mask

    def load_lidar_mask(self):
        if not os.path.exists(self.lidar_mask_path):
            self.lidar_mask = np.zeros(self.image.shape[:2])
            return self.lidar_mask
        else:
            self.lidar_range_mask = cv2.imread(self.lidar_range_mask_path, 1)
            self.lidar_mask = cv2.imread(self.lidar_mask_path, 1)
            combine = cv2.addWeighted(self.lidar_mask, 1, self.lidar_range_mask, 1, 0)
            self.lidar_mask = cv2.cvtColor(combine, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(self.lidar_mask, cv2.COLOR_BGR2GRAY)
            lidar_mask_alpha = np.zeros(gray.shape, dtype=np.float32)

            # gray = gray[gray > 0]
            lidar_mask_alpha[gray > 0] = 0.4
            lidar_mask_alpha = np.expand_dims(lidar_mask_alpha, axis=2)
            return self.lidar_mask, lidar_mask_alpha

    def save_mask(self, mask):
        cv2.imwrite(self.mask_path, 255 * mask.astype(np.uint8))


if __name__ == "__main__":

    def load_timestamp(root, log_name):
        radar_folder = root + "oxford/" + log_name + "/"
        timestamps_path = radar_folder + "radar.timestamps"
        radar_timestamps = np.loadtxt(
            timestamps_path, delimiter=" ", usecols=[0], dtype=np.int64
        )

        return radar_timestamps

    root = "/media/ee904/data/"
    log_name = "2019-01-10-11-46-21-radar-oxford-10k"
    data_root = (
        "/media/ee904/data/oxford/2019-01-10-11-46-21-radar-oxford-10k/RaMNet_data/"
    )
    radar_timestamps = load_timestamp(root, log_name)

    loader = DataLoader(data_root, radar_timestamps)
    loader.load_data(5)
    image = loader.load_image()
    mask = loader.load_mask()
    print("mask shape:", mask.shape, "image_shape", image.dtype)
    test_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    # loader.save_mask(test_mask)
    cv2.imshow("mask", mask)
    cv2.waitKey()
