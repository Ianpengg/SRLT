import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class CellClassification(nn.Module):
    def __init__(self, category_num=5):
        super(CellClassification, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        # x [bs*2, 32, 256, 256]
        x = F.relu(self.bn1(self.conv1(x))) # [bs*2, 32, 256, 256]
        x = self.conv2(x) # [bs*2, 5, 256, 256]

        return x


class StateEstimation(nn.Module):
    def __init__(self, motion_category_num=2):
        super(StateEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, motion_category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self, x):
        # x [bs*2, 32, 256, 256]
        x = F.relu(self.bn1(self.conv1(x))) # [bs*2, 32, 256, 256]
        #x = self.dropout(x)
        x = self.conv2(x) # [bs*2, 2, 256, 256]
        x = torch.sigmoid(x)
        return x


class MotionPrediction(nn.Module):
    def __init__(self, seq_len):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        # x [bs*2, 32, 256, 256]
        x = F.relu(self.bn1(self.conv1(x))) # [bs*2, 32, 256, 256]
        x = self.conv2(x) # [bs*2, 2*20, 256, 256]

        return x


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x


class STPN(nn.Module):
    def __init__(self, height_feat_size=13, use_temporal_info=True, num_past_frames=5, MCdrop=True):
        super(STPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=0.5)
        self.MCdrop = MCdrop 
        ### FRANK rm TCN ###
        if use_temporal_info:
            print('use temporal info')
            if num_past_frames >= 5:
                self.conv3d_1 = Conv3D(64, 64, kernel_size=(5, 1, 1), stride=1, padding=(0, 0, 0))
                self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
            if num_past_frames==3:
                self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
                self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
            if num_past_frames==2:
                print('num_past_frames==2')
                self.conv3d_1 = Conv3D(64, 64, kernel_size=(2, 1, 1), stride=1, padding=(0, 0, 0))
                self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
            else:
                print('no temporal info')
                self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
                self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        # x.shape: [bs*2, 5, 13, 256, 256]
        batch, seq, z, h, w = x.size()  #[batch, 5, 1, 256, 256]

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1)) # [bs*2*5, 13, 256, 256]  [batch*5, 1, 256, 256]
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x))) # [bs*2*5, 32, 256, 256]  [batch*5, 32, 256, 256]
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x))) # [bs*2*5, 32, 256, 256]  [batch*5, 32, 256, 256]

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))    #[batch*5,64, 128, 128) 
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w) 
       
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)
        
        if (self.MCdrop):
            # -- STC block 2
            x_2 = F.relu(self.bn2_1(self.conv2_1(self.dropout(x_1))))
            x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

            x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
            x_2 = self.conv3d_2(x_2)
            x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1
        
            # -- STC block 3
            x_3 = F.relu(self.bn3_1(self.conv3_1(self.dropout(x_2))))
            x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

            # -- STC block 4
            
            x_4 = F.relu(self.bn4_1(self.conv4_1(self.dropout(x_3))))
            x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))
        else:
            # -- STC block 2
            x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
            x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

            x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
            x_2 = self.conv3d_2(x_2)
            x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1
            # -- STC block 3
            x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
            x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

            # -- STC block 4
            x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
            x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))


        # -------------------------------- Decoder Path --------------------------------
        if (self.MCdrop):
            x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), self.dropout(x_3)), dim=1))))
        else:
            x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()
        if (self.MCdrop):
            x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), self.dropout(x_2)), dim=1))))
        else:
            x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))

        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x



class RaMNet(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, cell_category_num=2, height_feat_size=13,
                        use_temporal_info=True, num_past_frames=5, MCdropout=False):
        super(RaMNet, self).__init__()
        self.out_seq_len = out_seq_len

        #self.cell_classify = CellClassification(category_num=cell_category_num)
        #self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn = STPN(height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames, MCdrop=MCdropout)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w) # [bs*2, 5, 13, 256, 256]

        # Backbone network
        x = self.stpn(bevs) # [bs*2, 32, 256, 256]

        # Cell Classification head
        #cell_class_pred = self.cell_classify(x) # [bs*2, 2, 256, 256]

        # Motion State Classification head
        state_class_pred = self.state_classify(x) # [bs*2, 2, 256, 256]

        # Motion Displacement prediction
        #disp = self.motion_pred(x) # [bs*2, 20*2, 256, 256]
        #disp_ = disp.view(-1, 2, x.size(-2), x.size(-1)) # [20*bs*2, 2, 256, 256]

       
#        print('--- motionnet ---')
#        print(bevs.shape)
#        print(x.shape)
#        print(disp_.shape)
#        print(cell_class_pred.shape)
#        print(state_class_pred.shape)

#        torch.Size([bs*2, 5, 13, 256, 256])
#        torch.Size([bs*2, 32, 256, 256])
#        torch.Size([20*bs*2, 2, 256, 256])
#        torch.Size([bs*2, 5, 256, 256])
#        torch.Size([bs*2, 2, 256, 256])

        return state_class_pred

class RansacNet(nn.Module):
    def __init__(self):
        super(RansacNet, self).__init__()
        self.odom_from_disp = OdometryFromDisp()

    def forward(self, disp):
#        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#        ax.imshow(np.linalg.norm(disp[0].cpu().detach().numpy(), axis=0))
#        plt.show()
        odom = self.odom_from_disp(disp) # [1, 3]
        return odom

# For MGDA loss computation
class FeatEncoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(FeatEncoder, self).__init__()
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.stpn(bevs)

        return x


class MotionNetMGDA(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2):
        super(MotionNetMGDA, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

    def forward(self, stpn_out):
        # Cell Classification head
        cell_class_pred = self.cell_classify(stpn_out)

        # Motion State Classification head
        state_class_pred = self.state_classify(stpn_out)

        # Motion Displacement prediction
        disp = self.motion_pred(stpn_out)
        disp = disp.view(-1, 2, stpn_out.size(-2), stpn_out.size(-1))

        return disp, cell_class_pred, state_class_pred
