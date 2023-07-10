from torchvision import transforms as T
import torch
import torch.nn as nn
from PIL import Image

class Encode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self, x):
        return self.encode(x)

    
class Pool2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # in_ch x in_ch -> in_ch / 2 x in_ch / 2
        self.pooling = nn.Conv2d(in_channels,in_channels,
                                 kernel_size=4, stride=2, padding=1,
                                 padding_mode='reflect')
    
    def forward(self, x):
        return self.pooling(x)


class Decode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                      padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self, x):
        return self.decode(x)
    
    
class Unpool2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # in_ch x in_ch -> in_ch * 2 x in_ch * 2
        self.unpooling = nn.ConvTranspose2d(in_channels, in_channels,
                                            kernel_size=4, stride=2, padding=1)
    
    def forward(self, x, output_size):
        return self.unpooling(x, output_size=output_size)
    
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2)                              :
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=4, stride=stride, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.conv(x)
        
        
# U-net generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Downsampling
        
        # 256 -> 128
        self.encode0 = Encode(3, 64)
        self.pool0 = Pool2(64)
        
        # 128 -> 64
        self.encode1 = Encode(64, 128)
        self.pool1 = Pool2(128)
        
        # 64 -> 32
        self.encode2 = Encode(128, 256)
        self.pool2 = Pool2(256)
        
        # 32 -> 16 (8)
        self.encode3 = Encode(256, 512)
        self.pool3 = Pool2(512)
        
        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        
        # Upsampling
        
        # 16 -> 32
        self.unpool0 = Unpool2(512)
        self.decode0 = Decode(512, 256)
        
        # 32 -> 64
        self.unpool1 = Unpool2(256)
        self.decode1 = Decode(256, 128)
        
        # 64 -> 128
        self.unpool2 = Unpool2(128)
        self.decode2 = Decode(128, 64)
        
        # 128 -> 256 (128)
        self.unpool3 = Unpool2(64)
        self.decode3 = nn.Sequential(  # without last LeakyReLU
            nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True, padding_mode='reflect'),
            nn.InstanceNorm2d(3),
        )
        
    def forward(self, x):
        # encoder
        skip0 = self.encode0(x)
        e0 = self.pool0(skip0)

        skip1 = self.encode1(e0)
        e1 = self.pool1(skip1)

        skip2 = self.encode2(e1)
        e2 = self.pool2(skip2)

        skip3 = self.encode3(e2)
        e3 = self.pool3(skip3)

        # bottleneck
        b = self.bottleneck_conv(e3)

        # decoder
        d0 = self.decode0(torch.cat(
            [self.unpool0(b, output_size=skip3.size()), skip3], dim=1))
        d1 = self.decode1(torch.cat(
            [self.unpool1(d0, output_size=skip2.size()), skip2], dim=1))
        d2 = self.decode2(torch.cat(
            [self.unpool2(d1, output_size=skip1.size()), skip1], dim=1))
        d3 = self.decode3(torch.cat(
            [self.unpool3(d2, output_size=skip0.size()), skip0], dim=1))
        
        return d3
        
        
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.init = nn.Sequential(  # 256 -> 128
            nn.Conv2d(3*2, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, True),
        )
        self.conv = nn.Sequential(
            CNNBlock(64, 128),  # 128 -> 64
            CNNBlock(128, 256),  # 64 -> 32
            CNNBlock(256, 512),  # 32 -> 16
            CNNBlock(512, 512),  # 16 -> 8 (4)
            #CNNBlock(512, 512),  # 8 -> 4
        )
      
        self.pred = nn.Conv2d(512, 1, kernel_size=4, stride=1)
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        
        x = self.init(x)
        x = self.conv(x)
        return self.pred(x)
    
    
    
def load_model(model, path_d='/kaggle/input/pix2pix128/disriminator.pt', path_g='/kaggle/input/pix2pix128/generator.pt'):
    
    model["discriminator"].load_state_dict(torch.load(path_d, map_location='cpu')))
    model["generator"].load_state_dict(torch.load(path_g, map_location='cpu')))
