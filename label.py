import argparse
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rospy
import torch
import torch.nn as nn
from copy import deepcopy
from image import load_image
from camera_model import CameraModel
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as R
from models.model import RaMNet
from utils import oxford_utils
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# get from `rosrun tf tf_echo velodyne_left pseudo_lidar``
# pseudo_lidar_to_lidar_left_tf
pseudo_lidar_to_lidar_left_tf = np.zeros((4,4))
r = R.from_quat([-0.704, 0.710, 0.017, 0.013])
pseudo_lidar_to_lidar_left_tf[:3,:3] = r.as_matrix()
pseudo_lidar_to_lidar_left_tf[:3,3] = np.array([-0.449, 0.011, -0.001])
pseudo_lidar_to_lidar_left_tf[3,3] = 1

# get from `rosrun tf tf_echo velodyne_right pseudo_lidar``
# pseudo_lidar_to_lidar_right_tf
pseudo_lidar_to_lidar_right_tf = np.zeros((4,4))
r = R.from_quat([-0.705, 0.709, 0.014, 0.016])
pseudo_lidar_to_lidar_right_tf[:3,:3] = r.as_matrix()
pseudo_lidar_to_lidar_right_tf[:3,3] = np.array([0.449, -0.000, 0.001])
pseudo_lidar_to_lidar_right_tf[3,3] = 1

# # #get from `rosrun tf tf_echo radar pseudo_lidar``
# # [Some Bug exist]  Still on testing
pseudo_lidar_to_radar_tf = np.zeros((4,4))
r = R.from_quat([0.707, 0.707, 0.018, 0.018])
pseudo_lidar_to_radar_tf[:3,:3] = r.as_matrix()
pseudo_lidar_to_radar_tf[:3,3] = np.array([0.093, -0.012, 0.280])
pseudo_lidar_to_radar_tf[3,3] = 1

# pseudo_lidar_to_ins_tf
pseudo_lidar_to_ins_tf = np.zeros((4,4))
r = R.from_quat([0.709, 0.705, 0.019, -0.010])
pseudo_lidar_to_ins_tf[:3,:3] = r.as_matrix()
pseudo_lidar_to_ins_tf[:3,3] = np.array([0.010, -1.102, -1.464])
pseudo_lidar_to_ins_tf[3,3] = 1

def motion_comp_overlap(radar_curr, radar_past):
    """
    Return the overlap image between current radar(t) and past radar(t-1) 
    params: thres -> to adjust the overlapped region threshold 
    """ 
    thres = 0.15

    copy_radar_1 = deepcopy(radar_curr)
    copy_radar_2 = deepcopy(radar_past)

    copy_radar_1[copy_radar_1 < thres] = 0 
    copy_radar_1[copy_radar_1 > thres] = 1 

    copy_radar_2[copy_radar_2 < thres] = 0 
    copy_radar_2[copy_radar_2 > thres] = 1 

    temp_image = np.zeros((256, 256, 3))
    temp_image[:, :, 1] = copy_radar_2.squeeze(axis=2)

    temp_layer_2 = np.zeros((256, 256, 3))
    temp_layer_2[:, :, 2] = copy_radar_1.squeeze(axis=2)
    return temp_image + temp_layer_2

def read_point_cloud(path):
    return np.fromfile(path, dtype=np.float32).reshape(4, -1) # (x,y,z,I) * N

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def file_to_id(str_):
    idx = str_[find(str_,'/')[-1]+1:str_.find('.npy')]
    idx = int(idx)
    return idx



class MotionLabelInterface:
    def __init__(self, fig, ax, ax2, labeled_data_root, data_root, filename, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fig = fig
        self.ax = ax
        self.ax2 = ax2
        self.press = None
        self.filename = filename
        self.data_root = data_root
        self.models_dir = data_root + '/robotcar-dataset-sdk/models'
        self.modelpath = args.modelpath
        self.seq_dirs = [os.path.join(labeled_data_root, f) for f in os.listdir(labeled_data_root) if os.path.isfile(os.path.join(labeled_data_root, f))]
        self.seq_dirs.sort(key=file_to_id)
        
        # Label configs ex: visualization controls
        self.idx = 730
        self.old_idx = -1
        self.paint_size = 0
        self.mcdrop = args.mcdrop
        self.show_lidar = False
        self.show_stereo = False
        self.show_mono_rear = False
        self.show_mono_left = False
        self.show_mono_right = False
        self.label_thres = 0.13
        #
        rospy.init_node('talker', anonymous=True)
        self.lidar_pub = rospy.Publisher('/lidarpoint', PointCloud2, queue_size=100)
        self.load_model()
        self.connect()
        self.data_handler() 

    def load_model(self, ):
        self.model = RaMNet(out_seq_len=1,
                            motion_category_num=2,
                            cell_category_num=2, 
                            height_feat_size=1, 
                            use_temporal_info=True, 
                            num_past_frames=2,
                            MCdropout=True)
        try:
            checkpoint = torch.load(self.modelpath)
            self.model.load_state_dict(checkpoint['model_state_dict'], False)
            self.model = self.model.to(self.device)
            print("Loaded pretrained model from {}".format(self.modelpath))
        except FileNotFoundError:
            print("Failed to load model... check the model path is correct")

    def data_handler(self,):

        path = self.data_root + self.filename + '/'
        path_cam = self.data_root + self.filename + '/'
        
        # Load radar ts
        radar_folder = path + 'radar/'
        timestamps_path = path + 'radar.timestamps'
        self.radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
        if self.show_lidar:
            self.lidar_l_folder = path+'velodyne_left/'
            self.lidar_l_timestamps_path = path+'velodyne_left.timestamps'
            self.lidar_l_timestamps = np.loadtxt(self.lidar_l_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
    
            self.lidar_r_folder = path+'velodyne_right/'
            self.lidar_r_timestamps_path = path+'velodyne_right.timestamps'
            self.lidar_r_timestamps = np.loadtxt(self.lidar_r_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
         
        if self.show_stereo:
             self.stereo_folder = path_cam + 'stereo/centre/'
             stereo_timestamps_path = path_cam+'stereo.timestamps'
             self.stereo_timestamps = np.loadtxt(stereo_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
             self.stereo_model = CameraModel(self.models_dir, self.stereo_folder)
             
        if self.show_mono_rear:
             self.mono_rear_folder = path_cam + 'mono_rear/'
             mono_rear_timestamps_path = path_cam + 'mono_rear.timestamps'
             self.mono_rear_timestamps = np.loadtxt(mono_rear_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
             self.mono_rear_model = CameraModel(self.models_dir, self.mono_rear_folder)

        if self.show_mono_left:
             self.mono_left_folder = path_cam + 'mono_left/'
             mono_left_timestamps_path = path_cam + 'mono_left.timestamps'
             self.mono_left_timestamps = np.loadtxt(mono_left_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
             self.mono_left_model = CameraModel(self.models_dir, self.mono_left_folder)
             
        if self.show_mono_right:
             self.mono_right_folder = path_cam + 'mono_right/'
             mono_right_timestamps_path = path_cam + 'mono_right.timestamps'
             self.mono_right_timestamps = np.loadtxt(mono_right_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
             self.mono_right_model = CameraModel(self.models_dir, self.mono_right_folder)

    def label(self, ):

        while self.idx >= 0:
            dims = np.array([256, 256])
            if self.old_idx != self.idx:
                   
                self.gt_file_path = self.seq_dirs[self.idx]
                with open(self.gt_file_path, 'rb') as f:
                    gt_data_handle = np.load(f, allow_pickle=True)
                self.gt_dict = gt_data_handle.item()

                radar_idx = file_to_id(self.gt_file_path)
                radar_timestamp = self.radar_timestamps[int(radar_idx)]       

                if self.show_lidar:
                    lidar_r_idx, lidar_r_timestamp = oxford_utils.get_sync(radar_timestamp, self.lidar_r_timestamps)
                    
                    lidar_l_idx, lidar_l_timestamp = oxford_utils.get_sync(radar_timestamp, self.lidar_l_timestamps)
                    curr_lidar_l_t = oxford_utils.UnixTimeToSec(lidar_l_timestamp)
                    curr_lidar_r_t = oxford_utils.UnixTimeToSec(lidar_r_timestamp)
                    
                    lidar_l_filename = self.lidar_l_folder + str(lidar_l_timestamp) + '.bin'
                    if not os.path.isfile(lidar_l_filename):
                        print("Could not find radar example: {}".format(lidar_l_filename))
                    pc_left = read_point_cloud(lidar_l_filename).T # (N x 4)
                    pc_left[:,3] = 1
                    new_pc_left = np.dot(pseudo_lidar_to_lidar_left_tf, pc_left.T).T

                    lidar_r_filename = self.lidar_r_folder + str(lidar_r_timestamp) + '.bin'
                    if not os.path.isfile(lidar_r_filename):
                        print("Could not find radar example: {}".format(lidar_r_filename))
                    pc_right = read_point_cloud(lidar_r_filename).T # (N x 4)
                    pc_right[:,3] = 1
                    new_pc_right = np.dot(pseudo_lidar_to_lidar_right_tf, pc_right.T).T
                    new_pc = np.concatenate((new_pc_left, new_pc_right), axis=0)
                    header = Header()
                    header.stamp = rospy.get_rostime() #rospy.Time.from_sec(curr_t)
                    header.frame_id = "radar"
                    fields = [PointField('x', 0, PointField.FLOAT32, 1),
                            PointField('y', 4, PointField.FLOAT32, 1),
                            PointField('z', 8, PointField.FLOAT32, 1)
                            ]
                    pc_msg = point_cloud2.create_cloud(header, fields, new_pc[:,:3])
                    self.lidar_pub.publish(pc_msg)
                

                if self.show_stereo:
                    stereo_idx, stereo_timestamp = oxford_utils.get_sync(radar_timestamp, self.stereo_timestamps)
                    filename = self.stereo_folder + str(stereo_timestamp) + '.png'
                    stereo_image_data = load_image(filename, model=None)
                    stereo_image_data = cv2.cvtColor(stereo_image_data, cv2.COLOR_RGB2BGR)

                if self.show_mono_rear:
                    mono_rear_idx, mono_rear_timestamp = oxford_utils.get_sync(radar_timestamp, self.mono_rear_timestamps)
                    filename = self.mono_rear_folder + str(mono_rear_timestamp) + '.png'
                    rear_image_data = load_image(filename, model=self.mono_rear_model)
                    rear_image_data = cv2.cvtColor(rear_image_data, cv2.COLOR_RGB2BGR)

                if self.show_mono_left:
                    mono_left_idx, mono_left_timestamp = oxford_utils.get_sync(radar_timestamp, self.mono_left_timestamps)
                    filename = self.mono_left_folder + str(mono_left_timestamp) + '.png'
                    left_image_data = load_image(filename, model=self.mono_left_model)
                    left_image_data = cv2.cvtColor(left_image_data, cv2.COLOR_RGB2BGR)

                if self.show_mono_right:
                    mono_right_idx, mono_right_timestamp = oxford_utils.get_sync(radar_timestamp, self.mono_right_timestamps)
                    filename = self.mono_right_folder + str(mono_right_timestamp) + '.png'
                    right_image_data = load_image(filename, model=self.mono_right_model)
                    right_image_data = cv2.cvtColor(right_image_data, cv2.COLOR_RGB2BGR)
                
                if self.show_stereo:
                    cv2.namedWindow('stereo_image_data',cv2.WINDOW_NORMAL)
                    cv2.imshow('stereo_image_data', stereo_image_data)
                if self.show_mono_rear: 
                    cv2.namedWindow('rear_image_data',cv2.WINDOW_NORMAL)
                    cv2.imshow('rear_image_data', rear_image_data)
                if self.show_mono_left:
                    cv2.namedWindow('left_image_data',cv2.WINDOW_NORMAL)
                    cv2.imshow('left_image_data', left_image_data)
                if self.show_mono_right:
                    cv2.namedWindow('right_image_data',cv2.WINDOW_NORMAL)
                    cv2.imshow('right_image_data', right_image_data)
                cv2.waitKey(1)

                
                raw_radars = list()
                raw_radar_list = list()
                for j in range(2):
                    raw_radars.append(np.expand_dims(self.gt_dict['raw_radar_' + str(j)], axis=2))
                raw_radars = np.stack(raw_radars, 0).astype(np.float32)
                raw_radar_list.append(raw_radars)
                
                self.raw_radar_data = np.stack(raw_radar_list, 0)
                self.raw_radar_data = torch.tensor(self.raw_radar_data).to(self.device) 


                raw_radar_curr = raw_radars[0].squeeze()
                
                self.viz_raw_radar = np.stack((raw_radar_curr, raw_radar_curr, raw_radar_curr), axis=2) * 2

                self.pixel_moving_map_ = self.gt_dict['gt_moving']
                self.pixel_moving_map_[self.pixel_moving_map_ > 0] = 1

            viz_motion = np.zeros((256, 256, 3))
            viz_motion[:, :, 0] = self.pixel_moving_map_
            overlapped_radar = motion_comp_overlap(raw_radars[0], raw_radars[1])

            # load the center car image
            car_image = plt.imread('./car.png')
            offset_image = OffsetImage(car_image, zoom=0.1, alpha=0.5)
            box = AnnotationBbox(offset_image, [127, 127], frameon=False, zorder=999)
            

            self.ax2.clear()
            self.ax2.axis('off')
            self.ax2.set_aspect('equal')
            self.ax2.title.set_text(str(self.idx) + "  " + self.gt_file_path)
            self.ax2.title.set_color('w')  
            self.ax2.imshow(np.clip((overlapped_radar + self.viz_raw_radar * 2) / 2., 0, 1.), cmap="jet")

            self.ax.clear()
            self.ax.axis('off')
            self.ax.set_aspect('equal')
            self.ax.title.set_text(self.idx)
            self.ax.title.set_color('w')
            self.ax.add_artist(box)
            self.ax.imshow(np.clip((viz_motion + self.viz_raw_radar * 2) / 2., 0, 1.))
            self.connect()

            self.old_idx = self.idx
            
            while plt.waitforbuttonpress() != True:
                x = 0
        plt.show()


    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release) 
        self.cidkey = self.fig.canvas.mpl_connect(
             "key_press_event", self.on_key)
        
    def on_key(self, event):
        global pixel_radar_map_, ax 
        paint_size = self.paint_size
        #print('you pressed', event.key, event.x, event.y, event.xdata, event.ydata)
        if event.key == 'd':
            print("Go to next frame")
            self.gt_dict['gt_moving'] = self.pixel_moving_map_.astype(np.bool_)
            with open(self.gt_file_path, 'wb') as f:
                np.save(f, arr=self.gt_dict)
            print('[save]', self.gt_file_path)
            self.idx += 1
        if event.key == 'a':
            print("Go to previous frame")
            self.gt_dict['gt_moving'] = self.pixel_moving_map_.astype(np.bool_)
            #np.save(self.gt_file_path, arr=self.gt_dict)
            with open(self.gt_file_path, 'wb') as f:
                np.save(f, arr=self.gt_dict)
            
            print('[save]', self.gt_file_path)
            if self.idx-1 < 0:
                pass
            else:
                self.idx -= 1
        if event.key == '1':
            xdata = int(event.xdata)
            ydata = int(event.ydata)
            for i in range(-paint_size, paint_size+1):
                for j in range(-paint_size, paint_size+1):

                    x = xdata + i
                    y = ydata + j
                    if self.viz_raw_radar[y, x][0] >= self.label_thres:
                        self.pixel_moving_map_[y, x] = 1

        if event.key == '2':
            xdata = int(event.xdata)
            ydata = int(event.ydata)
            for i in range(-paint_size, paint_size+1):
                for j in range(-paint_size, paint_size+1):
                    #print(i,j)
                    x = xdata + i
                    y = ydata + j
                    if self.viz_raw_radar[y, x][0] >= self.label_thres:
                        self.pixel_moving_map_[y, x] = 0
                        
        if event.key == '3':
            xdata = int(event.xdata)
            ydata = int(event.ydata)
            for i in range(-paint_size, paint_size+1):
                for j in range(-paint_size, paint_size+1):
                    x = xdata + i
                    y = ydata + j
                    self.pixel_moving_map_[y, x] = 1
                    
        if event.key == '4':
            xdata = int(event.xdata)
            ydata = int(event.ydata)
            for i in range(-paint_size, paint_size+1):
                for j in range(-paint_size, paint_size+1):
                    x = xdata + i
                    y = ydata + j
                    self.pixel_moving_map_[y, x] = 0
                    
        if event.key == 'i':
        
            if self.mcdrop:
                self.model.apply(lambda module: setattr(module, 'training', True) if isinstance(module, nn.Dropout2d) else None)
                self.model.apply(lambda module: setattr(module, 'training', False) if isinstance(module, nn.BatchNorm2d) else None)
                self.model.apply(lambda module: setattr(module, 'training', False) if isinstance(module, nn.BatchNorm3d) else None)

            self.model.eval()
            with torch.no_grad():
                motion_pred = self.model(self.raw_radar_data)
            motion_pred_numpy = motion_pred.cpu().numpy()
            infer_motion_pred_p = motion_pred_numpy[0, 0, :, :]
            infer_motion_pred = infer_motion_pred_p > 0.5
            infer_motion_pred[infer_motion_pred > 0 ] = 1
            infer_motion_pred[infer_motion_pred <= 0 ] = 0
            self.pixel_moving_map_ = infer_motion_pred
            
        # To fill the empty region inside the contour 
        if event.key == 'j':
            kernel = np.ones((5,5),np.uint8)
            self.pixel_moving_map_ = cv2.morphologyEx(self.pixel_moving_map_.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Clear annotations on currnet image 
        if event.key == ' ':
            #self.pixel_moving_map_[:,:] = 0
            print('[clear axis]')
            
        # Change the brush size
        if event.key == 'z':
            print("Set the painter size => 0")
            self.paint_size = 0
        if event.key == 'x':
            print("Set the painter size => 1")
            self.paint_size = 1
        if event.key == 'c':
            print("Set the painter size => 2")
            self.paint_size = 2
        if event.key == 'v':
            print("Set the painter size => 3")
            self.paint_size = 3
        if event.key == 'b':
            print("Set the painter size => 5")
            self.paint_size = 5
        if event.key == 'escape':
            self.disconnect()
            exit()


    def on_press(self, event):
        print(event.xdata, event.ydata)
        """Check whether mouse is over us; if so, store some data."""
        self.press = (event.xdata, event.ydata)


    def on_release(self, event):
        """Clear button press information."""
        self.press = None
        self.fig.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.fig.figure.canvas.mpl_disconnect(self.cidpress)
        self.fig.figure.canvas.mpl_disconnect(self.cidrelease)
        self.fig.figure.canvas.mpl_disconnect(self.cidkey)



