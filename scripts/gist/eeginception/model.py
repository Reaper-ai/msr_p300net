import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGInceptionPyTorch(nn.Module):
    """
    EEG-Inception Network (Dağ et al. Table II) in PyTorch.
    """
    def __init__(self, num_channels=8, time_samples=128, filters_per_branch=8, dropout_rate=0.25, n_classes=1):
        super().__init__()
        self.conv_t64 = nn.Conv2d(1, filters_per_branch, (1, 64), padding=(0, 32), bias=False)
        self.conv_t32 = nn.Conv2d(1, filters_per_branch, (1, 32), padding=(0, 16), bias=False)
        self.conv_t16 = nn.Conv2d(1, filters_per_branch, (1, 16), padding=(0, 8), bias=False)
        self.bn1 = nn.BatchNorm2d(filters_per_branch * 3)
        
        total_b1_filters = filters_per_branch * 3
        self.conv_spatial = nn.Conv2d(total_b1_filters, total_b1_filters * 2, kernel_size=(num_channels, 1), groups=total_b1_filters, bias=False)
        self.bn2 = nn.BatchNorm2d(total_b1_filters * 2)
        self.pool1 = nn.AvgPool2d((4, 1))
        self.drop1 = nn.Dropout2d(dropout_rate)
        
        spatial_out_channels = total_b1_filters * 2
        self.conv_b2_16 = nn.Conv2d(spatial_out_channels, filters_per_branch, (1, 16), padding=(0, 8), bias=False)
        self.conv_b2_8  = nn.Conv2d(spatial_out_channels, filters_per_branch, (1, 8), padding=(0, 4), bias=False)
        self.conv_b2_4  = nn.Conv2d(spatial_out_channels, filters_per_branch, (1, 4), padding=(0, 2), bias=False)
        self.bn3 = nn.BatchNorm2d(filters_per_branch * 3)
        self.pool2 = nn.AvgPool2d((2, 1))
        self.drop2 = nn.Dropout2d(dropout_rate)
        
        b2_out_channels = filters_per_branch * 3
        self.conv_out1 = nn.Conv2d(b2_out_channels, b2_out_channels // 2, (8, 1), padding=(4, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(b2_out_channels // 2)
        self.pool3 = nn.AvgPool2d((2, 1))
        self.drop3 = nn.Dropout2d(dropout_rate)
        
        self.conv_out2 = nn.Conv2d(b2_out_channels // 2, b2_out_channels // 4, (4, 1), padding=(2, 0), bias=False)
        self.bn5 = nn.BatchNorm2d(b2_out_channels // 4)
        self.pool4 = nn.AvgPool2d((2, 1))
        self.drop4 = nn.Dropout2d(dropout_rate)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, time_samples)
            out = self.forward_features(dummy.permute(0, 1, 3, 2))
            flat_size = out.reshape(1, -1).shape[1]
        self.fc = nn.Linear(flat_size, n_classes)

    def forward_features(self, x):
        t64 = self.conv_t64(x)
        t32 = self.conv_t32(x)
        t16 = self.conv_t16(x)
        b1_concat = torch.cat([t64, t32, t16], dim=1)
        out = F.elu(self.bn1(b1_concat))
        
        out = self.conv_spatial(out)
        out = F.elu(self.bn2(out))
        out = self.pool1(out)
        out = self.drop1(out)
        
        b2_16 = self.conv_b2_16(out)
        b2_8  = self.conv_b2_8(out)
        b2_4  = self.conv_b2_4(out)
        b2_concat = torch.cat([b2_16, b2_8, b2_4], dim=1)
        out = F.elu(self.bn3(b2_concat))
        out = self.pool2(out)
        out = self.drop2(out)
        
        out = self.conv_out1(out)
        out = F.elu(self.bn4(out))
        out = self.pool3(out)
        out = self.drop3(out)
        
        out = self.conv_out2(out)
        out = F.elu(self.bn5(out))
        out = self.pool4(out)
        out = self.drop4(out)
        return out

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        features = self.forward_features(x)
        return self.fc(features.reshape(features.size(0), -1))
