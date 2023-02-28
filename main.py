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



motionInterface = MotionLabelInterface(dataset_root, data_root, filename, args)
motionInterface.label()
