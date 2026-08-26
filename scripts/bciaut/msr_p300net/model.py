import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeEEGAugmenter(nn.Module):
    """
    On-the-fly PyTorch-based data augmenter for EEG trials, matching the implementation in main.py.
    """
    def __init__(self, p=0.5, noise_std=0.05, scale_range=(0.9, 1.1), max_shift=2, drop_prob=0.1):
        super().__init__()
        self.p = p
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.max_shift = max_shift
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training: 
            return x
        b, ch, t = x.shape
        x_aug = x.clone()

        if torch.rand(1).item() < self.p:
            scales = torch.empty(b, 1, 1, device=x.device).uniform_(*self.scale_range)
            x_aug = x_aug * scales
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise
        if torch.rand(1).item() < self.p:
            mask = (torch.rand(b, ch, 1, device=x.device) > self.drop_prob).float()
            x_aug = x_aug * mask
        if torch.rand(1).item() < self.p:
            shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
            if shift != 0:
                x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                if shift > 0: 
                    x_aug[..., :shift] = 0
                else: 
                    x_aug[..., shift:] = 0
        return x_aug

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
    Multi-Scale Refined Network (MSR_P300NET) for BCIAUT Dataset.
    """
    def __init__(self, num_channels=8, input_time_samples=140):
        super().__init__()
        self.block1 = MultiScaleStemBlock(in_channels=1, spatial_channels=12, depth_multiplier=2, eeg_channels=num_channels)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.block2 = DSTemporalBlock(channels=24, kernel_size=5, dilation=2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop2 = nn.Dropout(0.25)
        self.block3 = FeatureFusionBlock(in_channels=24, out_channels=32, kernel_size=5)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2))
        self.drop3 = nn.Dropout(0.30)

        with torch.no_grad():
            x = torch.zeros(1, 1, num_channels, input_time_samples)
            x = self.drop1(self.pool1(self.block1(x)))
            x = self.drop2(self.pool2(self.block2(x)))
            x = self.drop3(self.pool3(self.block3(x)))
            self.flat_dim = x.numel()

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
    Applies Max-Norm constraints on the weights of spatial convolution and linear classifier layers.
    """
    with torch.no_grad():
        w1 = model.block1.conv_s.weight
        norm1 = w1.norm(2, dim=(1, 2, 3), keepdim=True)
        w1.mul_(torch.clamp(norm1, max=1.0) / (norm1 + 1e-8))
        
        wd = model.classifier[-1].weight
        normd = wd.norm(2, dim=1, keepdim=True)
        wd.mul_(torch.clamp(normd, max=0.25) / (normd + 1e-8))
