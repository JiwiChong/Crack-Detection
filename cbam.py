import torch
import torch.nn as nn
import torch.nn.functional as F


class Channel_Attention(nn.Module):
    def __init__(self, input_feature_dim, channel_dim=None, stride=1):
        super(Channel_Attention, self).__init__()
        self.stride = stride
        self.mlp1 = nn.Linear(channel_dim, int(channel_dim/2))
        self.mlp2 = nn.Linear(int(channel_dim/2), channel_dim)

        self.maxpool = nn.MaxPool2d(input_feature_dim, stride=self.stride)
        self.avgpool = nn.AvgPool2d(input_feature_dim, stride=self.stride)
        # Fully connected layers

    def forward(self, input_feature):
        maxp_out = self.maxpool(input_feature)
        avgp_out = self.avgpool(input_feature)

        maxp_mlp1_out = self.mlp1(maxp_out.squeeze(3).squeeze(2))
        maxp_mlp2_out = F.sigmoid(self.mlp2(maxp_mlp1_out))

        avgp_mlp1_out = self.mlp1(avgp_out.squeeze(3).squeeze(2))
        avgp_mlp2_out = F.sigmoid(self.mlp2(maxp_mlp1_out))

        maxp_mlp2_out_ = maxp_mlp2_out.unsqueeze(2).unsqueeze(3)
        avgp_mlp2_out_ = avgp_mlp2_out.unsqueeze(2).unsqueeze(3)
        channel_attention = maxp_mlp2_out_ + avgp_mlp2_out_

        channel_att_out = torch.mul(input_feature, channel_attention)
        return channel_att_out

class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention, self).__init__()

        self.att_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, first_att_input):
        avg_pools = []
        max_pools = []
        for b in range(first_att_input.shape[0]):
            avg_pool = torch.mean(first_att_input[b,:,:,:], dim=0)
            max_pool = torch.max(first_att_input[b,:,:,:], dim=0).values
            avg_pools.append(avg_pool.unsqueeze(0))
            max_pools.append(max_pool.unsqueeze(0))

        avg_pool_layer = torch.stack(avg_pools)
        max_pool_layer = torch.stack(max_pools)
        atten_layer = torch.concatenate((avg_pool_layer, max_pool_layer), dim=1)

        convolution_out = F.sigmoid(self.att_conv(atten_layer))
        return convolution_out

class CBAM(nn.Module):
    def __init__(self, input_feature_dim=None, channel_dim=None, stride=1):
        super(CBAM, self).__init__()
        self.channel_att_module = Channel_Attention(input_feature_dim, channel_dim, stride)
        self.spatial_att_module = Spatial_Attention()

    def forward(self, input_feature):
        channel_att_out = self.channel_att_module(input_feature)
        spatial_att_out = self.spatial_att_module(channel_att_out)
        cbam_out = input_feature + spatial_att_out
        return cbam_out


    

