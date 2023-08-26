# SRLT: Scanning Radar Labelling Tool for annotating segmentation mask 

<img src="assets/main_gui.png?raw=true" width="720"/>

## About SRLT 

SRLT is a graphical scanning radar image annotation tool.

It's written in Python and uses Qt for its graphical interface.

**This tool currently only supports the Scanning Radar Image**   


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

### radar 
Scanning radar images (in cartesian form) which should be the same size in all files

### lidar mask
The corresponding LiDAR BEV images are stored in the "lidar_mask" folder. We first project the LiDAR data onto the Bird's Eye View (BEV) and ensure that the pixel resolution remains the same as the radar image.

### range mask
The range mask consists of multiple rings that indicate the range from the center of the radar image.

### radar.timestamps 
This timestamps file should store all filenames in the `radar` folder.

We have provide function that can extract the timestamps from the radar folder also the function that project LiDAR to BEV 

###

```
└── DATA_PATH
  ├── radar/
  | ├── 1547120787893007.jpg
  | ├── 1547120788140671.jpg
  | └── ...
  ├── lidar_mask/
  | ├── 1547120787893007.jpg
  | ├── 1547120788140671.jpg
  | └── ...
  |
  ├── range_mask.png
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
