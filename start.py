
import os 
#import torch
import numpy as np
import argparse

from PyQt5 import QtWidgets

from gui.controller import MainWindow_controller
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator



def load_timestamp(root, log_name):

    radar_folder = root + 'oxford/' + log_name + '/'
    timestamps_path = radar_folder + 'radar.timestamps'
    radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

    return radar_timestamps


if __name__ == '__main__':
    import sys
     
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root", type=str, help="specify the datapath")
    parser.add_argument("--checkpoint", type=str, default="./models/sam_vit_h_4b8939.pth", help="specify the checkpoint path")
    parser.add_argument("--model_type", type=str, default="vit_h", help="specify the model type")
    args = parser.parse_args()


    root = '/media/ee904/data/'
    log_name = '2019-01-10-11-46-21-radar-oxford-10k' 
    data_root = args.data_root
    radar_timestamps = load_timestamp(root, log_name)

    sam_checkpoint = args.checkpoint
    model_type = args.model_type

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    sam_controller = SamPredictor(sam)
    

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller(data_root, radar_timestamps, sam_controller)
    window.show()
    sys.exit(app.exec_())
