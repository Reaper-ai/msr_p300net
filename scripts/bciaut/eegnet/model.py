import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    Standard EEGNet architecture (Borra et al. / Lawhern et al.)
    Adapted to support both 3D [batch, 8, 140] and 4D [batch, 1, 8, 140] inputs.
    """
    def __init__(self, num_channels=8, time_samples=140, dropout_rate=0.25):
        super().__init__()
        # Temporal convolution
        self.conv_temp = nn.Conv2d(1, 8, (1, 65), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Spatial convolution (Depthwise)
        self.conv_depth = nn.Conv2d(8, 16, (num_channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout2d(dropout_rate)
        
        # Separable convolution
        self.conv_sep_depth = nn.Conv2d(16, 16, (1, 17), padding=(0, 8), groups=16, bias=False)
        self.conv_sep_point = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout2d(dropout_rate)
        
        # Calculate linear classifier input dimensions automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, time_samples)
            x = self.pool2(self.bn3(self.conv_sep_point(self.conv_sep_depth(self.drop1(self.pool1(self.bn2(self.conv_depth(F.elu(self.bn1(self.conv_temp(dummy)))))))))))
            flat_size = x.view(1, -1).shape[1]
        
        self.fc = nn.Linear(flat_size, 2, bias=True)
    
    def forward(self, x):
        # Auto-unsqueeze 3D inputs to 4D to match Conv2D requirements
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.drop1(self.pool1(F.elu(self.bn2(self.conv_depth(F.elu(self.bn1(self.conv_temp(x))))))))
        x = self.drop2(self.pool2(F.elu(self.bn3(self.conv_sep_point(self.conv_sep_depth(x))))))
        return self.fc(x.view(x.size(0), -1))
