# SRLT: Scanning Radar Labelling Tool for annotating segmentation mask 

SRLT is a graphical scanning radar image annotation tool.

It's written in Python and uses Qt for its graphical interface.

**This tool currently supports the Oxford Radar Robotcar Dataset, you can also modified it to use in custom dataset.**   

## Description
<img src="assets/main_gui.png?raw=true" width="720"/>


## Features
- [x] Radar image segmentation annotation.
- [x] Segment Anything Model(SAM) supported semi-auto annotation 


## Folder Structure
```
SRLT/
├── data/
│   ├── prepare_data.py
│   ├── gen_lidar_mask.py
│   └── mask2box.py
├── gui/
│   ├── utils/
│   │   ├── file_utils.py
│   │   ├── gui_utils.py
│   │   └── image_method.py
│   ├── interact/
│   │   ├── interaction.py
│   │   └── timer.py
│   ├── btn_controller.py
│   ├── controller.py
│   ├── inference_core.py
│   ├── shortcut.py
│   └── ui.py
├── models
├── itri.sh
├── oxford.sh
├── setup.py
└── start.py
```
## Getting Started

### Prerequisites

#### 1. Install Ian's toolbox library
> This is the custom library that contains multiple useful functions to deal with radar image , LiDAR publish, and transforms.. etc
```
pip install iantoolbox
```
#### 2. Install the Segment Anything Model(SAM)
Install the SAM for semi-auto annotatation
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
#### 2. Download the necessary files
For example : one **sequence** among Oxford Radar RobotCar Dataset.

In this annotation tool, the necessary files include:
- **Navtech CTS350-X Radar** - raw radar data 
- **Navtech CTS350-X Radar Optimised SE2 Odometry** -used to compensate the ego-motion
- (Optional) **Velodyne HDL-32E Left Pointcloud**- (used for lidar BEV reference) 
- (Optional) **Velodyne HDL-32E Right Pointcloud**- (used for lidar BEV reference) 

#### 3. Preprocess the required data
1. Generate the compensated $N$ radar scans (Current + $N\!-\!1$ historical scans)
```
/data/prepare_data.py
```

2. (Optional) Generate the LiDAR BEV images for reference
```
/data/gen_lidar_mask.py
```

The final folder structure for **data** will looks like 
```
data/
├── radar/
│   ├── xxxx.png
│   └── ...
├── radar_history/
│   ├── xxxx_1.png
│   ├── xxxx_2.png
│   └── ...
└── lidar_mask/
    └── xxxx.png
```

### Installation
1. Clone the repository
```sh
git clone https://github.com/Ianpengg/SRLT
```
2. Setup the environment

```sh
# Setup conda
conda create -n srlt python=3.7.5
conda activate srlt

# Install requirement
pip install -r requirements.txt
```

## Usage
Open the GUI
```
oxford.sh
```

## Reference 
Parts of the code have been adapted from the following repository:
https://github.com/ori-mrg/robotcar-dataset-sdk

## LICENSE
This project is licensed under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For the full license, please refer to the LICENSE file.
