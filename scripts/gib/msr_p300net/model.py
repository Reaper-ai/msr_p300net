import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleStemBlock(nn.Module):
    def __init__(self, in_channels=1, spatial_channels=12, depth_multiplier=2, eeg_channels=8):
        super().__init__()
        self.conv_t15 = nn.Conv2d(in_channels, spatial_channels // 3, (1, 15), padding=(0, 7), bias=False)
        self.conv_t31 = nn.Conv2d(in_channels, spatial_channels // 3, (1, 31), padding=(0, 15), bias=False)
        self.conv_t63 = nn.Conv2d(in_channels, spatial_channels // 3, (1, 63), padding=(0, 31), bias=False)
        self.bn_t = nn.BatchNorm2d(spatial_channels)
        out_spatial = spatial_channels * depth_multiplier
        self.conv_s = nn.Conv2d(spatial_channels, out_spatial, (eeg_channels, 1), groups=spatial_channels, bias=False)
        self.bn_s = nn.BatchNorm2d(out_spatial)

    def forward(self, x):
        t_concat = torch.cat([self.conv_t15(x), self.conv_t31(x), self.conv_t63(x)], dim=1)
        return F.elu(self.bn_s(self.conv_s(self.bn_t(t_concat))))

class DSTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=2):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1_dw = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, pad), dilation=(1, dilation), groups=channels, bias=False)
        self.conv1_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2_dw = nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, pad), dilation=(1, dilation), groups=channels, bias=False)
        self.conv2_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1_pw(self.conv1_dw(x))))
        out = self.bn2(self.conv2_pw(self.conv2_dw(out)))
        return F.elu(out + x)

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv_dw = nn.Conv2d(in_channels, in_channels, (1, kernel_size), padding=(0, pad), groups=in_channels, bias=False)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, (1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.skip_proj = nn.Conv2d(in_channels, out_channels, (1, 1), bias=False)

    def forward(self, x):
        out = self.conv_pw(self.conv_dw(x))
        return F.elu(self.bn(out) + self.skip_proj(x))

class MSR_P300NET(nn.Module):
    """
    Multi-Scale Refined Network (MSR_P300NET) for GIB Dataset.
    Assumes input of shape [batch, channels=8, time=128] or [batch, 1, channels=8, time=128].
    """
    def __init__(self, n_channels=8, n_samples=128):
        super().__init__()
        self.block1 = MultiScaleStemBlock(in_channels=1, spatial_channels=12, depth_multiplier=2, eeg_channels=n_channels)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(0.25)
        
        self.block2 = DSTemporalBlock(channels=24, kernel_size=5, dilation=2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop2 = nn.Dropout(0.25)
        
        self.block3 = FeatureFusionBlock(in_channels=24, out_channels=32, kernel_size=5)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.drop3 = nn.Dropout(0.30)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            d_out = self.drop1(self.pool1(self.block1(dummy)))
            d_out = self.drop2(self.pool2(self.block2(d_out)))
            d_out = self.drop3(self.pool3(self.block3(d_out)))
            self.flat_dim = d_out.numel()

        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(self.flat_dim, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.drop1(self.pool1(self.block1(x)))
        x = self.drop2(self.pool2(self.block2(x)))
        x = self.drop3(self.pool3(self.block3(x)))
        return self.classifier(torch.flatten(x, 1))

def apply_max_norm(model):
    """
    Applies strict Max-Norm weight constraints to model parameters in training loop.
    """
    with torch.no_grad():
        w1 = model.block1.conv_s.weight
        norm1 = w1.norm(2, dim=(1, 2, 3), keepdim=True)
        w1.mul_(torch.clamp(norm1, max=1.0) / (norm1 + 1e-8))
        
        wd = model.classifier[-1].weight
        normd = wd.norm(2, dim=1, keepdim=True)
        wd.mul_(torch.clamp(normd, max=0.25) / (normd + 1e-8))
