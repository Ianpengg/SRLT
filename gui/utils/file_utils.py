import numpy as np
import os
import sys
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]
    
def file_to_id(str_):
    idx = str_[find(str_,'/')[-1]+1:str_.find('.npy')]
    idx = int(idx)
    return idx


def load_data(file_path, id):
    path = file_path + str(id) + '.npy'
    data = np.load(path, allow_pickle=True)
    data = data.item()
    return data

def load_image(file_path, id):
    """ 
    Load image from file path and id
    And cast the float32 to uint8
    """
    
    data = load_data(file_path, id)
    image = data['raw_radar_0']
    image = (image * 255).astype(np.uint8)
    image = np.stack((image, image, image), axis=2)
    return image

def load_mask(file_path, id):
    data = load_data(file_path, id)
    mask = data['gt_moving']
    mask[mask > 0] = 1
    
    return mask
