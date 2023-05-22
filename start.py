
import os 
#import torch
import numpy as np
import argparse

from PyQt5 import QtWidgets

from gui.controller import MainWindow_controller





def load_timestamp(root, log_name):

    radar_folder = root + 'oxford/' + log_name + '/'
    timestamps_path = radar_folder + 'radar.timestamps'
    radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

    return radar_timestamps


if __name__ == '__main__':
    import sys
     
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root", type=str, help="specify the datapath")
    args = parser.parse_args()
    root = '/media/ee904/data/'
    log_name = '2019-01-10-11-46-21-radar-oxford-10k' 
    data_root = args.data_root
    radar_timestamps = load_timestamp(root, log_name)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller(data_root, radar_timestamps)
    window.show()
    sys.exit(app.exec_())
