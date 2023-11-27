# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

from models_DDA.PanNet_DDA import SDL_attention
from common.basic import *


class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, criterion, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if 'conv' in k and 'weight' in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = criterion + sum(regularizations)
        return loss


class FusionNet(nn.Module):
    def __init__(self, spectral_num=8, channel=32):
        super(FusionNet, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num

        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.SDL = SDL_attention(inplanes=channel, planes=channel)  # 新添加的
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.SDL,
            self.res3,
            self.res4
        )

    def forward(self, x, y):  # x= lms; y = pan

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output  # lms + outs


if __name__ == '__main__':
    lms = torch.randn([1, 8, 64, 64])
    pan = torch.randn([1, 1, 64, 64])
    ms = torch.randn([1, 8, 16, 16])
    model = FusionNet(8)
    x = model(lms, pan)
    print(x.shape)
