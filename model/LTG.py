import torch
import torch.nn as nn
import torch.nn.functional as F
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class MSFP(nn.Module):
    def __init__(self, n_feats):
        super(MSFP, self).__init__()
        # n_feats_1, n_feats_2, n_feats_3 = n_feats
        self.conv12 = conv1x1(n_feats, n_feats) # from 2 to  1
        self.conv13 = conv1x1(n_feats, n_feats)  # from 3 to 1

        self.conv21 = conv3x3(n_feats, n_feats, 2) # from 1 to 2
        self.conv23 = conv1x1(n_feats, n_feats)   # from 3 to 2

        self.conv31_1 = conv3x3(n_feats, n_feats, 2)
        self.conv31_2 = conv3x3(n_feats, n_feats, 2)
        self.conv32 = conv3x3(n_feats, n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*3, n_feats)
        self.conv_merge2 = conv3x3(n_feats*3, n_feats)
        self.conv_merge3 = conv3x3(n_feats*3, n_feats)

    def forward(self, x3, x2, x1):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21, x31), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12, x32), dim=1) ))
        x3 = F.relu(self.conv_merge3( torch.cat((x3, x13, x23), dim=1) ))
        
        return x3, x2, x1


class LTG(torch.nn.Module):
    def __init__(self, n_feats, num_res_blocks=3):
        super(LTG, self).__init__()
        self.n_feats = n_feats
        n_feats_1, n_feats_2, n_feats_3 = [64, 128, 256]
        self.num_res_blocks = [2] * num_res_blocks
        self.conv_head_1 = conv1x1(n_feats_1, n_feats) # or 3x3?
        self.conv_head_2 = conv1x1(n_feats_2, n_feats)
        self.conv_head_3 = conv1x1(n_feats_3, n_feats)
        self.num = len(self.num_res_blocks)
        self.RB1 = nn.ModuleList()
        self.RB2 = nn.ModuleList()
        self.RB3 = nn.ModuleList()
        self.CSFI = nn.ModuleList()
        for i in range(self.num):
            self.CSFI.append(MSFP(n_feats))
            self.RB1.append(nn.ModuleList())
            for _ in range(self.num_res_blocks[i]):
                self.RB1[i].append(ResBlock(in_channels=n_feats, out_channels=n_feats))
            self.RB2.append(nn.ModuleList())
            for _ in range(self.num_res_blocks[i]):
                self.RB2[i].append(ResBlock(in_channels=n_feats, out_channels=n_feats))
            self.RB3.append(nn.ModuleList())
            for _ in range(self.num_res_blocks[i]):
                self.RB3[i].append(ResBlock(in_channels=n_feats, out_channels=n_feats))
                
            
        self.conv_tail_1 = conv3x3(n_feats, n_feats_1)
        self.conv_tail_2 = conv3x3(n_feats, n_feats_2)
        self.conv_tail_3 = conv3x3(n_feats, n_feats_3)

    def forward(self, lr_lv1=None, lr_lv2=None, lr_lv3=None):
        f_lv1 = lr_lv1
        f_lv2 = lr_lv2
        f_lv3 = lr_lv3
        lr_lv1 = F.relu(self.conv_head_1(lr_lv1))
        lr_lv2 = F.relu(self.conv_head_2(lr_lv2))
        lr_lv3 = F.relu(self.conv_head_3(lr_lv3))

        for i in range(self.num):
            for j in range(self.num_res_blocks[i]):
                lr_lv1 = self.RB1[i][j](lr_lv1)
                lr_lv2 = self.RB2[i][j](lr_lv2)
                lr_lv3 = self.RB3[i][j](lr_lv3)
            lr_lv1, lr_lv2, lr_lv3 = self.CSFI[i](lr_lv1, lr_lv2, lr_lv3)
        lr_lv1 = self.conv_tail_1(lr_lv1)
        lr_lv2 = self.conv_tail_2(lr_lv2)
        lr_lv3 = self.conv_tail_3(lr_lv3)
        return lr_lv3+f_lv3, lr_lv2+f_lv2, lr_lv1+f_lv1


if __name__ == '__main__':
    model = LTG(n_feats=64,num_res_blocks=[2,2,2])
    print('# para: %d' % sum(param.numel()
                             for param in model.parameters() if param.requires_grad))
    x1 = torch.rand((2,64,160,160))
    x2 = torch.rand((2,128,80,80))
    x3 = torch.rand((2,256,40,40))
    x3, x2, x1 = model(x1, x2, x3)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)