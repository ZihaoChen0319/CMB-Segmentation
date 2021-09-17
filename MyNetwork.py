import torch
import torch.nn as nn
import math
import torch.nn.functional as nnf


def _initialize_weights(net):
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2. / n))
            if layer.bias is not None:
                layer.bias.data.zero_()
        elif isinstance(layer, nn.Conv3d) or isinstance(layer, nn.ConvTranspose3d):
            n = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2] * layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2. / n))
            if layer.bias is not None:
                layer.bias.data.zero_()
        elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            layer.weight.data.normal_(0, 0.01)
            layer.bias.data.zero_()


class ConvBnRelu3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(ConvBnRelu3D, self).__init__()
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        _initialize_weights(self.conv)
        self.bn = nn.BatchNorm3d(out_channel)
        _initialize_weights(self.bn)
        self.relu = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, in_channel, out_channel, is_deconv=True,
                 kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)):
        super(UpSampling, self).__init__()
        self.in_channels = in_channel

        if is_deconv:
            self.block = nn.Sequential(
                nn.ConvTranspose3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(out_channel), nn.LeakyReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                ConvBnRelu3D(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=(1, 1, 1), use_1x1conv=False):
        super(ResBlock, self).__init__()
        self.use_1x1conv = use_1x1conv
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        if self.use_1x1conv:
            self.conv3 = nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1), stride=stride, padding=(0, 0, 0))

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_1x1conv:
            x = self.conv3(x)
        return self.relu2(out + x)


class ResDecBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, is_deconv=True,
                 kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)):
        super(ResDecBlock, self).__init__()
        self.in_channels = in_channel

        if is_deconv:
            self.block = nn.Sequential(
                ResBlock(in_channel, mid_channel, use_1x1conv=True),
                nn.ConvTranspose3d(mid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                ResBlock(in_channel, mid_channel, use_1x1conv=True),
                ResBlock(mid_channel, out_channel, use_1x1conv=True),
            )

    def forward(self, x):
        return self.block(x)


class ScreenNet(nn.Module):
    def __init__(self, is_fc=False, in_channel=3, num_class=2):
        super(ScreenNet, self).__init__()
        self.is_fc = is_fc
        self.net = nn.Sequential(
            ConvBnRelu3D(in_channel, 64, kernel_size=(5, 5, 3), padding=(0, 0, 0)),
            ConvBnRelu3D(64, 64, kernel_size=(5, 5, 3), padding=(0, 0, 0)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ConvBnRelu3D(64, 128, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            ConvBnRelu3D(128, 128, kernel_size=(3, 3, 3), padding=(0, 0, 0))
        )

        if self.is_fc:
            self.final = nn.Sequential(
                nn.Linear(128 * 2 * 2 * 2, 256),
                nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                nn.Linear(256, num_class)
            )
        else:
            self.final = nn.Sequential(
                nn.Conv3d(128, 256, kernel_size=(2, 2, 2), padding=(0, 0, 0)),
                nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                nn.Conv3d(256, num_class, kernel_size=(1, 1, 1), padding=(0, 0, 0))
            )
        _initialize_weights(self.final)

    def forward(self, x):
        x = self.net(x)
        if self.is_fc:
            x = x.view(-1, 128 * 2 * 2 * 2)
            x = self.final(x)
        else:
            x = self.final(x).squeeze(4).squeeze(3).squeeze(2)
        return x


class DiscriNet(nn.Module):
    def __init__(self, in_channel=3, num_class=2):
        super(DiscriNet, self).__init__()
        self.net = nn.Sequential(
            ConvBnRelu3D(in_channel, 64, kernel_size=(5, 5, 3), padding=(0, 0, 0)),
            ConvBnRelu3D(64, 64, kernel_size=(5, 5, 3), padding=(0, 0, 0)),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ConvBnRelu3D(64, 128, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
            ConvBnRelu3D(128, 128, kernel_size=(3, 3, 3), padding=(0, 0, 0)),
        )

        self.final = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 4, 512), nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Linear(512, num_class))
        _initialize_weights(self.final)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 128 * 4 * 4 * 4)
        x = self.final(x)
        return x


class UNet3Stage(nn.Module):
    def __init__(self, in_channel=3, num_class=2):
        super(UNet3Stage, self).__init__()
        self.num_class = num_class

        self.enc_conv1 = nn.Sequential(
            ConvBnRelu3D(in_channel, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ConvBnRelu3D(32, 32))
        self.enc_conv2 = nn.Sequential(
            ConvBnRelu3D(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            ConvBnRelu3D(64, 64))
        self.enc_conv3 = nn.Sequential(
            ConvBnRelu3D(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            ConvBnRelu3D(128, 128))
        self.bottom = nn.Sequential(
            ConvBnRelu3D(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            ConvBnRelu3D(256, 256), UpSampling(256, 128))
        self.dec_conv3 = nn.Sequential(
            ConvBnRelu3D(256, 128), ConvBnRelu3D(128, 128), UpSampling(128, 64))
        self.dec_conv2 = nn.Sequential(
            ConvBnRelu3D(128, 64), ConvBnRelu3D(64, 64), UpSampling(64, 32))
        self.dec_conv1 = nn.Sequential(ConvBnRelu3D(64, 32), ConvBnRelu3D(32, 32))
        self.seg_out_layer = nn.Conv3d(32, num_class, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(enc1)
        enc3 = self.enc_conv3(enc2)
        dec3 = self.bottom(enc3)
        dec2 = self.dec_conv3(torch.cat([enc3, dec3], 1))
        dec1 = self.dec_conv2(torch.cat([enc2, dec2], 1))
        out_seg = self.dec_conv1(torch.cat([enc1, dec1], 1))
        out_seg = self.seg_out_layer(out_seg)
        out_seg = nnf.softmax(out_seg, dim=1)

        return out_seg

