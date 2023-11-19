# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# spatial-spectral domain attention learning(SDL)
class SDL_attention(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, mode="ch*sp"):
        super(SDL_attention, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.mode = mode

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

        # new
        # self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
        #                        bias=True)
        # todo conv改成kernel_size=1或5试试

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    # HR spatial attention
    def spatial_attention(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()

        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)

        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)

        mask_ch = self.sigmoid(context)

        return mask_ch

    # HR spectral attention
    def spectral_attention(self, x):

        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()

        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)

        mask_sp = self.sigmoid(context)

        return mask_sp

    def forward(self, x):
        mask_ch = self.spatial_attention(x)
        mask_sp = self.spectral_attention(x)
        if self.mode == "ch":
            out = x * mask_ch
        elif self.mode == "sp":
            out = x * mask_sp
        elif self.mode == "ch+sp":
            out = x * mask_ch + x * mask_sp
        elif self.mode == "ch*sp":
            out = x * mask_ch * mask_sp
        else:
            raise ValueError("DDA mode is unsupported")

        return out


# -------------Initialization----------------------------------------
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                # print("nn.Conv2D is initialized by variance_scaling_initializer")
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv1(x))  # Bsx32x64x64
        rs1 = self.conv2(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


# -----------------------------------------------------
class PanNet(nn.Module):
    def __init__(self, spectral_num=8, channel=32, mode="ch*sp", place=3):
        super(PanNet, self).__init__()

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
                                         padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.SDL = SDL_attention(inplanes=channel, planes=channel, mode=mode)  # 新添加的

        if place == 1:
            self.backbone = nn.Sequential(self.SDL, self.res1, self.res2, self.res3, self.res4)
        elif place == 3:
            self.backbone = nn.Sequential(self.res1, self.res2, self.SDL, self.res3, self.res4)
        elif place == 5:
            self.backbone = nn.Sequential(self.res1, self.res2, self.res3, self.res4, self.SDL)

        self.apply(init_weights)
        init_weights(self.backbone, self.deconv, self.conv1, self.conv3)  # state initialization, important!

    def forward(self, x, y):  # x= hp of ms; # Bsx8x16x16 y = hp of pan # Bsx1x64x64

        output_deconv = self.deconv(x)  # Bsx8x64x64
        input = torch.cat([output_deconv, y], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone! # Bsx32x64x64

        output = self.conv3(rs)  # Bsx8x64x64
        return output


# ----------------- End-Main-Part ------------------------------------


if __name__ == "__main__":
    # #########################测试数据 ################################
    hw = 16
    HW = 64
    pan = torch.rand([1, 1, HW, HW])
    ms = torch.rand([1, 8, hw, hw])

    _model = PanNet()

    out = _model(ms, pan)
    print(out.shape)

    # ##################################################################
