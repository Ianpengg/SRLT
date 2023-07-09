import os
import sys
import numpy as np
import argparse

from PyQt5 import QtWidgets

from gui.controller import MainWindow_controller
from segment_anything import sam_model_registry, SamPredictor
from utils.timestamp_loader import registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder_name", type=str, help="specify the folder name to labeled"
    )
    parser.add_argument(
        "-d", "--data_root", type=str, help="specify the path where the data locate"
    )
    parser.add_argument(
        "--dataset", type=str, default="oxford", help="specify which dataset is used"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./models/sam_vit_h_4b8939.pth",
        help="specify the checkpoint path",
    )
    parser.add_argument(
        "--model_type", type=str, default="vit_h", help="specify the model type"
    )
    parser.add_argument(
        "--patch_num", type=int, default="-1", help="specify which patch is used"
    )
    args = parser.parse_args()

    data_root = os.path.join(args.data_root, args.folder_name)

    timestamp_loader = f"{args.dataset}_timestamp_loader"

    radar_timestamps = registry.get(timestamp_loader)(args.data_root)

    sam_checkpoint = args.checkpoint
    model_type = args.model_type

    patch_num = args.patch_num
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_controller = SamPredictor(sam)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller(data_root, radar_timestamps, sam_controller, patch_num)
    window.show()
    sys.exit(app.exec_())
