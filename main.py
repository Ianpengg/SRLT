import argparse
import matplotlib.pyplot as plt
from label import MotionLabelInterface


dataset_root = '/data/Oxford/testing_data_v1/2019-01-10-12-32-52-radar-oxford-10k' 

filename = '2019-01-10-12-32-52-radar-oxford-10k'  # '2019-01-10-11-46-21-radar-oxford-10k'

data_root = '/data/Oxford/'

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', default='str', help='Model path that would be used')
parser.add_argument('--mcdrop', action='store_true', help='Whether to enable mcdropout model')
args = parser.parse_args()

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 12), facecolor=(0,0,0))
plt.tight_layout()

motionInterface = MotionLabelInterface(fig, ax, ax2, dataset_root, data_root, filename, args)
motionInterface.label()
