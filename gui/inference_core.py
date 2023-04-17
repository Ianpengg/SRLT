import torch
import numpy as np
from models.oxford_model import RaMNet  

class Inference_core:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = None

        # define model parameters
        self.out_seq_len = 1
        self.use_temporal_info = True
        self.num_past_frames = 2
        

    def set_model(self, model_path):
        self.model = RaMNet(out_seq_len=self.out_seq_len,
                            motion_category_num=2,
                            cell_category_num=2, 
                            height_feat_size=1, 
                            use_temporal_info=self.use_temporal_info, 
                            num_past_frames=self.num_past_frames,
                            MCdropout=False)
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'], False)
            self.model = self.model.to(self.device)
            
        except FileNotFoundError:
            print("Failed to load model... check the model path is correct")

        

         
    def inference(self, data):
        raw_radars = []
        # load raw radars
        for i in range(self.num_past_frames):
            raw_radars.append(np.expand_dims(data['raw_radar_' + str(i)], axis=2)) #[256-128:256+128, 256-128:256+128]

        raw_radars = np.stack(raw_radars, 0).astype(np.float32)
        raw_radars_list = []
        raw_radars_list.append(raw_radars)
        raw_radars = np.stack(raw_radars_list, 0)
        raw_radars = torch.tensor(raw_radars).to(self.device)
        # load car_mask
        if 'gt_moving' in data.keys():
            motion_gt = data['gt_moving']
        else: 
            motion_gt = np.zeros((256,256))
        motion_gt[motion_gt > 0] = 1


        self.model.eval()
        with torch.no_grad():
            if self.use_temporal_info:
                motion_pred = self.model(raw_radars)
        motion_pred_numpy = motion_pred.cpu().numpy()
        raw_radars = raw_radars.cpu().numpy()

        viz_motion_pred_p = motion_pred_numpy[0, 0, :, :] # foreground object
        viz_motion_pred = viz_motion_pred_p > 0.5
        viz_motion_pred[viz_motion_pred > 0] = 1
        viz_motion_pred[viz_motion_pred <= 0] = 0

        return viz_motion_pred