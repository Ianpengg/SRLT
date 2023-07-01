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
