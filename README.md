# SRLT: Scanning Radar Labelling Tool for annotating segmentation mask 

<img src="assets/main_gui.png?raw=true" width="720"/>

## About SRLT 

SRLT is a graphical scanning radar image annotation tool.

It's written in Python and uses Qt for its graphical interface.

**This tool currently only supports the Oxford Radar Robotcar Dataset**   


## Features

- [x] Intergrated the Segment Anything Model(SAM) from MetaAI
- [x] Image annotation with Brush and Box tools (using Box as a prompt for SAM).
- [x]  Image processing methods supported: brightness adjustment, contrast adjustment, and thresholding.
- [x] Mini-map for zooming in and out of sub-regions.
- [x] Auto-play button to show the consecutive frames.

<div>
    <img src="assets/mask_draw_undo.gif" alt="Mask Draw Undo" width="600"/>
    <img src="assets/brush_draw_undo.gif" alt="Brush Draw Undo" width="600"/>
    <img src="assets/image_process.gif" alt="Image Processing" width="600"/>
</div>


## Install
The code tested under `python==3.8`, `pytorch==1.12.1+cu113` and `torchvision==0.13.1+cu113` Please follow the instructions here to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install required packages:

```
cd SRLT; pip install -e .
```


## Data Preparation
Please follow the pre-processing pipeline in  [RaMNet](https://github.com/Ianpengg/RaMNet)
```
└── DATA_PATH
  ├── gt
  ├── radar
  ├── vo
  ├── processed/
  |   ├── radar/
  |   | ├── 1547120787893007.jpg
  |   | ├── 1547120788140671.jpg
  |   | └── ...
  |   ├── radar_history/
  |   |   ├── 1547120787893007_1.jpg
  |   |   ├── 1547120788140671_1.jpg
  |   |   └── ...
  |   ├── mask/
  |   |     ├── 1547120787893007.png
  |   |     ├── 1547120788140671.png
  |   |     └── ...
  |   ├── train.txt  # filename list for train 
  |   ├── val.txt  # filename list for validation 
  |   └── test.txt  # filename list for test 
  |
  ├── radar.timestamps
  ├── ...
```

## How to use 

```
python start.py -d DATA_PATH --checkpoint <your_sam_chackpoints> --model_type <sam_model_type>
```


## Reference 
Parts of the code have been adapted from the following repository:
https://github.com/ori-mrg/robotcar-dataset-sdk

## LICENSE
This project is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For the full license, please refer to the LICENSE file.
