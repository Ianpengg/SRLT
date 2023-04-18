import numpy as np
import os
import sys
def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]
    
def file_to_id(str_):
    idx = str_[find(str_,'/')[-1]+1:str_.find('.npy')]
    idx = int(idx)
    return idx



class DataLoader:
    def __init__(self, file_path, id):
        self.file_path = file_path
        self.id = id
        self.data = None
        self.image = None
        self.mask = None

    def is_valid(self):
        return os.path.exists(self.file_path + str(self.id) + '.npy')

    def load_data(self):
        path = self.file_path + str(self.id) + '.npy'
        try:
            data = np.load(path, allow_pickle=True)
            if data is not None:
                data = data.item()
                return data
        except FileNotFoundError:
            pass
        return None
    
    def load_image(self):
        if self.data is None:
            self.data = self.load_data()
        if self.data is not None and 'raw_radar_0' in self.data:
            image = self.data['raw_radar_0']
            image = (image * 255).astype(np.uint8)
            image = np.stack((image, image, image), axis=2)
            self.image = image
        else:
            self.image = None
        return self.image

    def load_mask(self):
        if self.data is None:
            self.data = self.load_data()
        if self.data is not None and 'gt_moving' in self.data:
            mask = self.data['gt_moving']
            mask[mask > 0] = 1
            self.mask = mask
        else:
            self.mask = None
        return self.mask
    
    def save_mask(self, mask):
        if self.data is None:
            self.data = self.load_data()
        if self.data is not None:
            self.data['gt_moving'] = mask.astype(np.float32)
            path = self.file_path + str(self.id) + '.npy'
            np.save(path, arr=self.data)
    