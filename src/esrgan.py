import torch
from torch.nn import Module, Conv2d, LeakyReLU, Sequential, PixelShuffle
from torch.nn.utils import spectral_norm

class DenseResidualBlock(Module):
    def __init__(self, in_c=64, gr_c=32):
        super().__init__()
        
        self.conv1 = Conv2d(in_channels=in_c, out_channels=gr_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=in_c+32, out_channels=gr_c, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2d(in_channels=in_c+64, out_channels=gr_c, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d(in_channels=in_c+96, out_channels=gr_c, kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv2d(in_channels=in_c+128, out_channels=in_c, kernel_size=3, stride=1, padding=1)
        
        self.leaky_relu = LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.leaky_relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.leaky_relu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        
        return x + x5 * 0.2

class RRDB(Module):
    def __init__(self, channels=64, gr_c=32):
        super().__init__()
        
        self.drb1 = DenseResidualBlock(in_c=channels, gr_c=gr_c)
        self.drb2 = DenseResidualBlock(in_c=channels, gr_c=gr_c)
        self.drb3 = DenseResidualBlock(in_c=channels, gr_c=gr_c)
        
    def forward(self, x):
        out = self.drb1(x)
        out = self.drb2(out)
        out = self.drb3(out)
        
        return x + out * 0.2

class RRDBNet(Module):

    def __init__(self, in_c=3, out_c=3, num_f=64, depth=23, gr_c=32, scale=4):
        super().__init__()
        
        self.scale = scale
        self.conv1 = Conv2d(in_channels=in_c, out_channels=num_f, kernel_size=3, stride=1, padding=1)
        self.rrdb = Sequential(*[RRDB(channels=num_f, gr_c=gr_c) for _ in range(depth)])
        self.conv2 = Conv2d(in_channels=num_f, out_channels=num_f, kernel_size=3, stride=1, padding=1)
        self.upsample = Sequential(
            Conv2d(in_channels=num_f, out_channels=num_f*4, kernel_size=3, stride=1, padding=1),
            PixelShuffle(upscale_factor=2),
            LeakyReLU(negative_slope=0.2, inplace=True),
            
            Conv2d(in_channels=num_f, out_channels=num_f*4, kernel_size=3, stride=1, padding=1),
            PixelShuffle(upscale_factor=2),
            LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.conv3 = Conv2d(in_channels=num_f, out_channels=num_f, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2d(in_channels=num_f, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        feature = self.conv1(x)
        rrdb_out = self.rrdb(feature)
        rrdb_out = self.conv2(rrdb_out)
        feature = feature + rrdb_out
        feature = self.upsample(feature)
        feature = self.conv3(feature)
        out = self.conv4(feature)
        return torch.clamp(out, 0, 1)



class Discriminator(Module):
    def __init__(self, in_c=3, num_f=64):
        super().__init__()
        
        def sn_conv(in_c, out_c, stride=1):
            return Sequential(
                spectral_norm(Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)),
                LeakyReLU(0.2, inplace=True)
            )
        
        self.first = Sequential(
            Conv2d(in_c, num_f, kernel_size=3, stride=1, padding=1),
            LeakyReLU(0.2, inplace=True)
        )
        
        self.features = Sequential(
            sn_conv(num_f, num_f, stride=2),
            sn_conv(num_f, num_f*2, stride=1),
            sn_conv(num_f*2, num_f*2, stride=2),
            sn_conv(num_f*2, num_f*4, stride=1),
            sn_conv(num_f*4, num_f*4, stride=2),
            sn_conv(num_f*4, num_f*8, stride=1),
            sn_conv(num_f*8, num_f*8, stride=2),
        )
        
        self.patch_classifier = Sequential(
            spectral_norm(Conv2d(num_f*8, num_f*8, kernel_size=3, stride=1, padding=1)),
            LeakyReLU(0.2, inplace=True),
            spectral_norm(Conv2d(num_f*8, 1, kernel_size=3, stride=1, padding=1))  # Raw logits
        )
        
    def forward(self, x):
        x = self.first(x)
        x = self.features(x)
        patches = self.patch_classifier(x)
        return patches