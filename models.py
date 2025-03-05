from __future__ import absolute_import, division, print_function
import torch.nn.functional as F
import numpy as np
import timm
import timm.models.layers as tlayers
import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
import einops

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
    
    
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class SkipBlocks(nn.Module):
    def __init__(self, in_channels, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(SkipBlocks, self).__init__()


        self.in_channels = in_channels
        self.inter_channels = in_channels# // 2

        conv = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g = nn.Sequential(
            conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.g[1].weight, 0)
        nn.init.constant_(self.g[1].bias, 0)
        self.W = nn.Sequential(
            conv(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)


    def forward(self, x1, x2):
        """_summary_

        Args:
            x1 (_type_): x1 input
            x2 (_type_): x2 input

        Returns:
            _type_: _description_
        """
        g_x1 = self.g(x1)
        W_y1 = self.W(g_x1)
        z1 = W_y1 + x1

        g_x2 = self.g(x2)
        W_y2 = self.W(g_x2)
        z2 = W_y2 + x2

        return z1,z2

# Inspired in H-Net code. Thanks to the authors.
class W_CA_Module(nn.Module):
    def __init__(self, in_channels, sizes, device, percentage=0.26, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True):
        """_summary_

        Args:
            in_channels (_type_): number of internal channels
            bn_layer (bool, optional): batch norm?. Defaults to True.
        """
        super(W_CA_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels

        #window creation. It is fixed for acceleration purposes.
        b, c, h, w = sizes

        bh = h
        max_w = int(w*percentage)

        slices = []
        for i in np.arange(w):
            # slice_1 = slices[i]
            slice_1 = torch.zeros(w)            
            slice_1[max(w-i-max_w,0):w-i] = torch.ones(min(w-i, max_w))
            slices.append(slice_1)

        r_l = torch.stack(slices, dim=0)
        r_l = torch.flip(r_l, dims=(0,))    

        for i in np.arange(w-1,-1,-1):
            slice_1 = r_l[i]
            slice_1[i:i+max_w] = torch.ones(min(w-i, max_w))
       
        if self.inter_channels is None:
            self.inter_channels = in_channels# // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        r_l = r_l.unsqueeze(0)
        r_l = r_l.repeat(bh,1,1)
        l_r = r_l

        self.l_r = l_r.bool().to(device)
        self.r_l = r_l.bool().to(device)        
        
        
        conv = nn.Conv2d
        bn = nn.BatchNorm2d

        if bn_layer:
            self.V = nn.Sequential(
                conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.V[1].weight, 0)
            nn.init.constant_(self.V[1].bias, 0)
            self.W = nn.Sequential(
                conv(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.V = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
            self.W = conv(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.Q = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.K = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)


    def forward(self, x1, x2):
        """_summary_

        Args:
            x1 : first input
            x2 : second output

        Returns:
            _type_: both attentions
        """
        [b,c,h,w] = x1.shape
        # print(b,c,h,w)  # 8 64 48 160

        #Query
        q_x1 = self.Q(x1).permute(0, 2, 3, 1).contiguous().view(-1, w, self.inter_channels) #Q
        q_x2 = self.Q(x2).permute(0, 2, 3, 1).contiguous().view(-1, w, self.inter_channels) #Q
        #Key
        k_x1 = self.K(x1).permute(0, 2, 1, 3).contiguous().view(-1, self.inter_channels, w) #K^t
        k_x2 = self.K(x2).permute(0, 2, 1, 3).contiguous().view(-1, self.inter_channels, w) #K^t
        #Value
        v_x1 = self.V(x1).permute(0, 2, 3, 1).contiguous().view(-1, w, self.inter_channels) #V
        v_x2 = self.V(x2).permute(0, 2, 3, 1).contiguous().view(-1, w, self.inter_channels) #V
        
        mask_l = []
        for idx in np.arange(b):
            mask_l.append(self.l_r)

        l_r = einops.rearrange(torch.stack(mask_l, dim=0), "B H W1 W2 -> (B H) W1 W2")#.shape
        r_l = l_r        
        
        kq_21 = torch.matmul(q_x1, k_x2) # (Q_x1 @ K_x2^T) V_x2 from path 2 to 1 
        kq_12 = torch.matmul(q_x2, k_x1) # (Q_x1 @ K_x2^T) V_x2 from path 2 to 1
        
        kq_21 = kq_21.masked_fill(~r_l, float("-inf"))
        kq_12 = kq_12.masked_fill(~r_l, float("-inf"))

        kq_21 = F.softmax(kq_21, dim=-1)
        kq_12 = F.softmax(kq_12, dim=-1)
        
        y1 = torch.matmul(kq_21, v_x2).contiguous() # softmax(QK^T)V
        y1 = y1.view(b, h, w, self.inter_channels).permute(0,3,1,2).contiguous() # to (b,c,h,w)

        W_y1 = self.W(y1) #(convolution)
        z1 = W_y1 + x1

        y2 = torch.matmul(kq_12, v_x1).contiguous()
        y2 = y2.view(b, h, w, self.inter_channels).permute(0,3,1,2).contiguous()
        W_y2 = self.W(y2)
        z2 = W_y2 + x2

        return z1,z2
        
        


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvnextCAEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, sizes, device, percentage=0.26):
        super(ConvnextCAEncoder, self).__init__()

        self.num_ch_enc = np.array([3, 64, 64, 128, 256, 512])

        convnext = timm.create_model("convnext_pico", pretrained=True, features_only=True)
        c1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=3)
        norm = tlayers.LayerNorm2d(64)
        c2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1,1), stride=(1,1))
        act = nn.GELU()
        c3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), stride=(1,1))
        
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        stem_0 = nn.Sequential(c1,
                               norm,
                               c2,
                               act,
                               c3,
                               maxpool)

        convnext.stem_0 = stem_0

        self.encoder = convnext
        
        sizes_1 = 2, 64, int(sizes[1]//4), int(sizes[2]//4)
        sizes_2 = 2, 128, int(sizes[1]//8), int(sizes[2]//8)
        sizes_3 = 2, 256, int(sizes[1]//16), int(sizes[2]//16)        

        self.ca_1 = W_CA_Module(in_channels=self.num_ch_enc[2], sizes=sizes_1, device=device, percentage=percentage)
        self.ca_2 = W_CA_Module(in_channels=self.num_ch_enc[3], sizes=sizes_2, device=device, percentage=percentage)
        self.ca_3 = W_CA_Module(in_channels=self.num_ch_enc[4], sizes=sizes_3, device=device, percentage=percentage)


    def forward(self, input_image0, input_image1):
        self.features0 = []
        self.features1 = []
        x0 = (input_image0 - 0.45) / 0.225 #default from monodepth2
        x1 = (input_image1 - 0.45) / 0.225 #default from monodepth2

        x0 = self.encoder.stem_0[0](x0)
        x0 = self.encoder.stem_0[1](x0)
        x0 = self.encoder.stem_0[2](x0)
        x0 = self.encoder.stem_0[3](x0)
        x0 = self.encoder.stem_0[4](x0)
        
        x1 = self.encoder.stem_0[0](x1)
        x1 = self.encoder.stem_0[1](x1)
        x1 = self.encoder.stem_0[2](x1)
        x1 = self.encoder.stem_0[3](x1)
        x1 = self.encoder.stem_0[4](x1)

        self.features0.append(x0) # to decoder (pos 0)
        self.features1.append(x1) # to decoder (pos 0)

        x0 = self.encoder.stem_0[-1](x0)
        x1 = self.encoder.stem_0[-1](x1)
        x0 = self.encoder.stem_1(x0)
        x1 = self.encoder.stem_1(x1)

        self.features0.append(self.encoder.stages_0(x0)) # to decoder (pos 1)
        self.features1.append(self.encoder.stages_0(x1)) # to decoder (pos 1)

        y01, y11 = self.ca_1(self.features0[-1], self.features1[-1])
        self.features0.append(self.encoder.stages_1(y01)) # to decoder (pos 2)
        self.features1.append(self.encoder.stages_1(y11)) # to decoder (pos 2)

        y02, y12 = self.ca_2(self.features0[-1], self.features1[-1])
        self.features0.append(self.encoder.stages_2(y02)) # to decoder (pos 3)
        self.features1.append(self.encoder.stages_2(y12)) # to decoder (pos 3)

        y03, y13 = self.ca_3(self.features0[-1], self.features1[-1])
        self.features0.append(self.encoder.stages_3(y03)) # to decoder (pos 4)
        self.features1.append(self.encoder.stages_3(y13)) # to decoder (pos 4)

        return [self.features0, self.features1]


class IDEP_Skip_Dual(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4)):
        super(IDEP_Skip_Dual, self).__init__()

        self.scales = scales
        self.num_ch_enc = num_ch_enc
        # print('self.num_ch_enc-------------', self.num_ch_enc)        # [ 64  64 128 256 512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        self.mix_conv = OrderedDict()
        # pooling
        self.pool = OrderedDict()

        num_ch=[64, 64, 128, 256, 512]
        for i in range(len(num_ch)):
            num_ch_in = 2*num_ch[i]
            num_ch_out = num_ch[i]
            self.mix_conv[('0', i)] = nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1, bias=False)
            self.mix_conv[('1', i)] = nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        ##### SkipBlocks
        for i in range(4, 1, -1):
            self.convs[('skip', i)] = SkipBlocks(in_channels=self.num_ch_dec[i])

        ############# Decoder branch 0 #############
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv0", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            if i!=0 and i<4:
                num_ch_in = self.num_ch_enc[i]+num_ch_out
                self.convs[("upconv0", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            elif i==0:
                self.convs[("upconv0", i, 1)] = ConvBlock(80, num_ch_out)                
            else:
                num_ch_in = num_ch_out
                self.convs[("upconv0", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
                
        ############# Decoder branch 1 #############
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv1", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            if i!=0 and i<4:
                num_ch_in = self.num_ch_enc[i]+num_ch_out
                self.convs[("upconv1", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            elif i==0:
                self.convs[("upconv1", i, 1)] = ConvBlock(80, num_ch_out)                
            else:
                num_ch_in = num_ch_out
                self.convs[("upconv1", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv0", s)] = Conv3x3(self.num_ch_dec[s], 1)
            self.convs[("dispconv1", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.middle = nn.ModuleList(list(self.mix_conv.values()))
        
    def forward(self, features):
        self.outputs = {}

        features0 = [self.relu(self.mix_conv[('0', i)](torch.cat([features[0][i], features[1][i]], 1))) for i in range(len(features[0]))]
        features1 = [self.relu(self.mix_conv[('1', i)](torch.cat([features[0][i], features[1][i]], 1))) for i in range(len(features[0]))]
        
        mid1 = features0
        mid2 = features1

        # features0, features1 = features

        x0 = features0[-1]
        x1 = features1[-1]        
        for i in range(4, -1, -1):

            x0 = self.convs[("upconv0", i, 0)](x0)  # x0 torch.Size([6,20])
            x1 = self.convs[("upconv1", i, 0)](x1)
                        
            if i<4:
                x0 = torch.cat((x0, features0[i]), 1)  #   # x0 torch.Size([6,20])
                x1 = torch.cat((x1, features1[i]), 1)  

                x0 = self.convs[("upconv0", i, 1)](x0)  # x0 torch.Size([6,20])                
                x1 = self.convs[("upconv1", i, 1)](x1)  
            elif i==4:
                x0 = self.convs[("upconv0", i, 1)](x0)  # x0 torch.Size([6,20])
                x1 = self.convs[("upconv1", i, 1)](x1)  

            if i >= 2:
                x0, x1 = self.convs[('skip', i)](x0, x1)

            x0 = upsample(x0)  # x0 torch.Size([12,40])            
            x1 = upsample(x1)  
            
            if i in self.scales:
                self.outputs[("disp0", i)] = self.sigmoid(self.convs[("dispconv0", i)](x0))
                self.outputs[("disp1", i)] = self.sigmoid(self.convs[("dispconv1", i)](x1))

        return self.outputs#, mid1, mid2      
    
    
    

class IDEP_Skip(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4)):
        super(IDEP_Skip, self).__init__()

        self.scales = scales
        self.num_ch_enc = num_ch_enc
        # print('self.num_ch_enc-------------', self.num_ch_enc)        # [ 64  64 128 256 512]
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        self.mix_conv = OrderedDict()
        # pooling
        self.pool = OrderedDict()

        num_ch=[64, 64, 128, 256, 512]
        for i in range(len(num_ch)):
            num_ch_in = 2*num_ch[i]
            num_ch_out = num_ch[i]
            self.mix_conv[('0', i)] = nn.Conv2d(num_ch_in, num_ch_out, kernel_size=3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        ##### SkipBlocks
        for i in range(4, 1, -1):
            self.convs[('skip', i)] = SkipBlocks(in_channels=self.num_ch_dec[i])

        ############# Decoder branch 0 #############
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[i]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv0", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            if i!=0 and i<4:
                num_ch_in = self.num_ch_enc[i]+num_ch_out
                self.convs[("upconv0", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            elif i==0:
                self.convs[("upconv0", i, 1)] = ConvBlock(80, num_ch_out)                
            else:
                num_ch_in = num_ch_out
                self.convs[("upconv0", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv0", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.ModuleList(list(self.mix_conv.values()) + list(self.convs.values()))

    def forward(self, features):
        self.outputs = {}

        features0 = [self.relu(self.mix_conv[('0', i)](torch.cat([features[0][i], features[1][i]], 1))) for i in range(len(features[0]))]

        # features0, features1 = features

        x0 = features0[-1]
        for i in range(4, -1, -1):

            x0 = self.convs[("upconv0", i, 0)](x0)  # x0 torch.Size([6,20])
                        
            if i<4:
                x0 = torch.cat((x0, features0[i]), 1)  #   # x0 torch.Size([6,20])

                x0 = self.convs[("upconv0", i, 1)](x0)  # x0 torch.Size([6,20])                
            elif i==4:
                x0 = self.convs[("upconv0", i, 1)](x0)  # x0 torch.Size([6,20])

            if i >= 2:
                x0, x0 = self.convs[('skip', i)](x0, x0)

            x0 = upsample(x0)  # x0 torch.Size([12,40])            
            
            if i in self.scales:
                self.outputs[("disp0", i)] = self.sigmoid(self.convs[("dispconv0", i)](x0))

        return self.outputs        
    
    
