#!/usr/bin/env python3
# @file   datasets.py
# @author Chen Yi Peng [ian01050.ee10@nycu.edu.tw]
# Licensed under the Apache License


import numpy as np
import os
import cv2
import lightning as L
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Manager
from ramnet.datasets.augmentation import ImageAugmentor
from ramnet.models import models

class OxfordSeqModule(L.LightningDataModule):
    """A Pytorch Lightning module for Oxford Radar RobotCar data"""

    def __init__(self, cfg):
        """Method to initizalize the KITTI dataset class
        Args:
          cfg: config dict

        Returns:
          None
        """
        super(OxfordSeqModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        dataset_name = self.cfg["TRAIN"]["DATASET"]
        ########## Point dataset splits
        if dataset_name == "oxford":
            train_set = OxfordDataset(self.cfg, split="train")
            val_set = OxfordDataset(self.cfg, split="val")
            test_set = OxfordDataset(self.cfg, split="test")
        elif dataset_name == "radiate":
            train_set = RadiateDataset(self.cfg, split="train")
            val_set = RadiateDataset(self.cfg, split="val")
            test_set = RadiateDataset(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=self.cfg["DATA"]["SHUFFLE"],
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        print("worker", self.cfg["DATA"]["NUM_WORKER"])
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)
        # for batch_idx, (data, target) in enumerate(self.train_iter):
        # data is a batch of input data
        # target is a batch of labels for the input data
        # print(
        #     "Batch {} - Data: {}, Target: {}".format(
        #         batch_idx, data.shape, target.shape
        #     )
        # )
        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


class OxfordDataset(Dataset):
    def __init__(self, cfg, split):
        """
        This dataloader loads multiple sequences for a keyframe, for computing the spatio-temporal consistency losses

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        num_past_frames: The number of past frames within a BEV sequence
        num_future_frames: The number of future frames within a BEV sequence. Default: 20
        num_category: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """

        cache_size = 10000

        self.cfg = cfg
        self.root_dir = os.environ.get("DATA")

        if self.root_dir is None:
            raise ValueError("The dataset root is None. Should specify its value.")

        ### load the data list of each split
        self.split = split

        if self.split not in ["train", "test", "val"]:
            raise Exception("Split must be train/val/test")

        #  extract the idx and frame_ts from .txt file save as a list
        with open(os.path.join(self.root_dir, f"{split}.txt"), "r") as f:
            self.frame_list = [
                image_info.strip().split(" ") for image_info in f.readlines()
            ]

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "train"
        if self.augment:
            self.augmentor = ImageAugmentor()

        self.radar_path = os.path.join(self.root_dir, "radar")
        self.radar_history_path = os.path.join(self.root_dir, "radar_history")
        self.mask_path = os.path.join(self.root_dir, "mask")

        self.num_past_frames = self.cfg["TRAIN"]["NUM_PAST_FRAMES"]
        self.dataset_size = len(self.frame_list) - 1
        self.frame_list = self.frame_list[1:]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        self.frame_idx, self.frame_ts = self.frame_list[idx]

        # aggregate the radar and history radar
        raw_radars = list()
        for i in range(self.num_past_frames):
            if i == 0:
                radar = cv2.imread(
                    os.path.join(self.radar_path, self.frame_ts + ".png"), 0
                )
                radar = (radar / 255).astype(np.float32)
                radar = np.expand_dims(radar, axis=2)

            if i > 0:
                radar = cv2.imread(
                    os.path.join(self.radar_history_path, self.frame_ts + f"_{i}.png"),
                    0,
                )
                radar = (radar / 255).astype(np.float32)

                radar = np.expand_dims(radar, axis=2)

            raw_radars.append(radar)

        moving_mask = cv2.imread(
            os.path.join(self.mask_path, self.frame_ts + ".png"), 0
        )
        moving_mask = moving_mask
        # cv2.imshow("mask", moving_mask)
        # cv2.waitKey()
        moving_mask = (moving_mask / 255).astype(np.float32)  # normalize to [0, 1]

        if self.augment:
            augment_result = self.augmentor(raw_radars[0], raw_radars[1], moving_mask)
            raw_radars[0] = augment_result["image"]
            raw_radars[1] = augment_result["image1"]
            moving_mask = augment_result["mask"]

        dims = raw_radars[0].shape[:2]
        # print(dims)
        gt_moving = np.zeros((dims[0], dims[1], 2))
        gt_moving[:, :, 0] = moving_mask  # moving
        gt_moving[:, :, 1] = np.logical_not(moving_mask)  # static

        raw_radars = np.stack(raw_radars, 0).astype(np.float32)
        # )  # (2, 256, 256, 1) = ( num_past_frams, h, w, 1)

        gt_moving = np.expand_dims(
            gt_moving, axis=0
        )  # (1, 256, 256, 2) = (-1, h, w, category)

        return raw_radars, gt_moving


class RadiateDataset(Dataset):
    def __init__(self, cfg, split):
        """
        This dataloader loads multiple sequences for a keyframe, for computing the spatio-temporal consistency losses

        dataset_root: Data path to the preprocessed sparse nuScenes data (for training)
        future_frame_skip: Specify to skip how many future frames
        voxel_size: The lattice resolution. Should be consistent with the preprocessed data
        area_extents: The area extents of the processed LiDAR data. Should be consistent with the preprocessed data
        num_past_frames: The number of past frames within a BEV sequence
        num_future_frames: The number of future frames within a BEV sequence. Default: 20
        num_category: The number of object categories (including the background)
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """

        cache_size = 10000

        self.cfg = cfg
        self.root_dir = os.environ.get("DATA")

        if self.root_dir is None:
            raise ValueError("The dataset root is None. Should specify its value.")

        ### load the data list of each split
        self.split = split

        if self.split not in ["train", "test", "val"]:
            raise Exception("Split must be train/val/test")

        #  extract the idx and frame_ts from .txt file save as a list
        with open(os.path.join(self.root_dir, f"{split}.txt"), "r") as f:
            self.frame_list = [
                image_info.strip().split(" ") for image_info in f.readlines()
            ]

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "train"
        if self.augment:
            self.augmentor = ImageAugmentor()

        self.radar_path = os.path.join(self.root_dir, "radar")
        self.radar_history_path = os.path.join(self.root_dir, "radar_history")
        self.mask_path = os.path.join(self.root_dir, "mask")

        self.num_past_frames = self.cfg["TRAIN"]["NUM_PAST_FRAMES"]
        self.dataset_size = len(self.frame_list)
        # self.frame_list = self.frame_list[1:]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        self.frame_idx, self.frame_ts = self.frame_list[idx]
        str_format = "{:06d}"
        # aggregate the radar and history radar
        raw_radars = list()
        for i in range(self.num_past_frames):
            if i == 0:
                radar = cv2.imread(
                    os.path.join(self.radar_path, str(self.frame_ts) + ".png"),
                    0,
                )
                radar = (radar / 255).astype(np.float32)
                radar = np.expand_dims(radar, axis=2)

            if i > 0:
                radar = cv2.imread(
                    os.path.join(
                        self.radar_history_path,
                        str(self.frame_ts) + f"_{i}.png",
                    ),
                    0,
                )
                radar = (radar / 255).astype(np.float32)

                radar = np.expand_dims(radar, axis=2)

            raw_radars.append(radar)

        moving_mask = cv2.imread(
            os.path.join(self.mask_path, str(self.frame_ts) + ".png"), 0
        )
        if moving_mask is None:
            moving_mask = np.zeros((radar.shape[:2]))
        moving_mask = moving_mask
        # cv2.imshow("mask", moving_mask)
        # cv2.waitKey()
        moving_mask = (moving_mask / 255).astype(np.float32)  # normalize to [0, 1]

        if self.augment:
            augment_result = self.augmentor(raw_radars[0], raw_radars[1], moving_mask)
            raw_radars[0] = augment_result["image"]
            raw_radars[1] = augment_result["image1"]
            moving_mask = augment_result["mask"]

        dims = raw_radars[0].shape[:2]
        # print(dims)
        gt_moving = np.zeros((dims[0], dims[1], 2))
        gt_moving[:, :, 0] = moving_mask  # moving
        gt_moving[:, :, 1] = np.logical_not(moving_mask)  # static

        raw_radars = np.stack(raw_radars, 0).astype(np.float32)
        # shape : ( 2, 256, 256, 1)
        # raw_radars = np.expand_dims(
        #     raw_radars, axis=0
        # )  # (1, 2, 256, 256) = (-1, num_past_frams, h, w)

        gt_moving = np.expand_dims(
            gt_moving, axis=0
        )  # (1, 256, 256, 2) = (-1, h, w, category)

        return raw_radars, gt_moving


if __name__ == "__main__":
    import yaml
    import argparse
    import torch
    import ramnet
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", type=str, default=None, help="the pre-trained model weight")
    
    args = parser.parse_args()

    cfg = {}
    cfg["TRAIN"] = {}
    cfg["TRAIN"]["NUM_PAST_FRAMES"] = 2 
    cfg["TRAIN"]["AUGMENTATION"] = False
    batchs = RadiateDataset(cfg=cfg, split="val")
    data_loader = DataLoader(batchs,  batch_size=1, shuffle=False)
    
    cfg = torch.load(args.weight)["hyper_parameters"]
    ckpt = torch.load(args.weight)
    model = models.MOSNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model.eval()
    model.freeze()

    OUTPUT_PATH = os.path.join(os.getcwd(), "output")
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    OUTPUT_PATH = os.path.join(OUTPUT_PATH, "mask")
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)



    for idx, data in enumerate(data_loader):
        raw_radars, gt_moving = data
        out = model(raw_radars.to("cuda"))
        #torch.Size([1, 2, 256, 256, 1]) torch.Size([1, 1, 256, 256, 2])

        motion_gt = gt_moving.cpu().detach().numpy()[0, 0, :, :, 0]
        raw_radar = raw_radars.cpu().detach().numpy()
        motion_pred = out.cpu().detach().numpy()[0, 0, :, :]
        motion_pred[motion_pred >= 0.5] = 1
        motion_pred[motion_pred < 0.5] = 0

        mask_color = np.zeros((*motion_pred.shape[:2], 3))
        mask_color[:, :, 2] = motion_pred * 255

        raw_radar = raw_radar[0,0,:,:,0]
        radar_img = np.stack((raw_radar, raw_radar, raw_radar), axis=2)


        mask_filename = os.path.join(OUTPUT_PATH, f"{batchs.frame_list[idx][1]}.png")
        #print(mask_filename)
        #print(np.max(motion_pred.flatten()))

        cv2.imwrite(mask_filename, (motion_pred*255).astype(np.uint8))
        #cv2.imshow("mass", mask_color)
        #cv2.imshow("radar", raw_radar)
        #cv2.imshow("radar_2", raw_radar[0,1, :, :, 0])
        #cv2.imshow("radar-over", raw_radar[0,0,:,:,0] + raw_radar[0,1, :, :, 0])
        #cv2.imshow("mask", motion_pred)
        cv2.imshow("overlap", 2*radar_img + mask_color*0.5)
        cv2.waitKey(10)

