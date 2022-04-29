# steer angle prediction model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import math

class BaseCNN(nn.Module):
    def __init__(self,type_='teacher'):
        super(BaseCNN, self).__init__()
        self.type = type_
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(32),
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),

            nn.Dropout(0.25)

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.InstanceNorm2d(128),

            nn.Dropout(0.25)

        )
        self.layer4 = nn.Sequential(
            nn.Linear(16*16*128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.layer5 = nn.Linear(1024, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer4(out)
        out2 = self.layer5(out)
        if self.type == 'teacher':
            return out2
        else:
            return out2, out

class Nvidia(nn.Module):
    def __init__(self, type_='teacher'):
        super(Nvidia, self).__init__()
        self.type = type_
        # 3*66*200
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 24*31*98
        self.layer2 = nn.Sequential(
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 36*14*47
        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 48*5*22
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 64*3*20
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True), 
        )
        # 64*1*8
        self.layer6 = nn.Sequential(
            #nn.Linear(64*1*18, 1164),
            nn.Linear(64*9*9, 1164),

            nn.ReLU(inplace=True),
            nn.Linear(1164, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
        )
        self.layer7 = nn.Linear(10, 1)
        
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer6(out)
        out2 = self.layer7(out)
        if self.type == 'teacher':
            return out2
        else:
            return out2, out

class Vgg16(nn.Module):
    def __init__(self, pretrained=False, type_='teacher'):
        super(Vgg16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        self.type = type_
        if pretrained:
            for parma in self.model.parameters():
                parma.requires_grad = False
        self.clf_layer1 = nn.Sequential(
            nn.BatchNorm1d(25088),
            nn.Linear(25088, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        #self.clf_layer1 = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.clf_layer2 = nn.Linear(1024, 1)

    
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.clf_layer1(x)
        out2 = self.clf_layer2(out)

        if self.type == 'teacher':
            return out2
        else:
            return out2, out


def build_vgg16(pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    if pretrained:
        for parma in model.parameters():
            parma.requires_grad = False    
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(25088),
        nn.Linear(25088, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )
    return model

def weight_init(m):
# 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
# m中的weight，bias其实都是Variable，为了能学习参数以及后向传播
        m.weight.data.fill_(1)
        m.bias.data.zero_()

# device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# net = BaseCNN()
# net = net.to(device)
# summary(net, (3,128,128))

class SteeringAngleRegressor(torch.nn.Module):
    def __init__(self, width, height, sequence_input=False, num_hidden=256):
        super().__init__()
        self.width = width
        self.height = height
        self.sequence_input = sequence_input
        self.feature_extractor = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=16, stride=1),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=16, stride=2),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=8, stride=2))
        
        
        self.input_dim = 50
        self.n_layers = 2
        self.n_hidden = num_hidden
        self.num_direction = 1
        dropout_p = 0
        assert self.num_direction in [1, 2]
        
        self.lstm = torch.nn.LSTM(input_size=self.input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
        self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 50)
        self.fc2 = torch.nn.Linear(50, 3) #angle, speed, torque
        
    def forward(self, x):
        if self.sequence_input:
            #input dim is [batch x time x channel x height x width]
            fe = torch.cat([self.feature_extractor(x[:, i, :, :, :]).unsqueeze(1) for i in range(x.size(1))], dim=1)
            fe = fe.view(fe.size(0), fe.size(1), fe.size(2)*fe.size(3)*fe.size(4))
            out, (hn, cn) = self.lstm(fe)
            return self.fc2(F.relu(self.fc1(out)))[:, :, 0]
        else:
            # input dim is [batch x channel x height x width]
            # [batch x channel x height x width]
            fe = self.feature_extractor(x)  # [batch x out_channels x some_height x some_width]
            fe = fe.view(fe.size(0), 1, fe.size(1)*fe.size(2)*fe.size(3)) # deal with this as if its time length is 1
            out, (hn, cn) = self.lstm(fe) # [batch x 1 x 256]
            return self.fc2(F.relu(self.fc1(out))).squeeze(dim=1)[:, :1] #[batch x 1 x 3] ->[batch]



# Model codes from 
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = torch.sigmoid(self.outc(x))
        return logits




class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)