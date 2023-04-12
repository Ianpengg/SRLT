
import os 
#import torch
import numpy as np

from PyQt5 import QtWidgets

from gui.controller import MainWindow_controller


def load_images(image_folder):
    frame_list = []
    for i, data in enumerate(image_folder):
        raw_radar = np.load(image_folder[i], allow_pickle=True)
        raw_radar = raw_radar.item()['raw_radar_0']
        #print(i)
        temp_img = np.stack((raw_radar, raw_radar, raw_radar), axis=2)
        frame_list.append(temp_img)
    frames = np.stack(frame_list, axis=0)
    return frames
if __name__ == '__main__':
    import sys

    #data_root = '/media/ee904/Data_stored/ramnet_data/training_data_v1/2019-01-10-11-46-21-radar-oxford-10k/'
    data_root = '/media/ee904/Data_stored/ramnet_data/testing_data_v1/2019-01-10-12-32-52-radar-oxford-10k/'
 
    data_list = os.listdir(data_root)
    data_list = [data_root + i for i in data_list]
    print(len(data_list))
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller(data_list)
    window.show()
    sys.exit(app.exec_())
