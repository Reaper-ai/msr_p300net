import torch
import torch.nn as nn
import torch.nn.functional as F

class P3CNET(nn.Module):
    """
    P3CNET Model (Dağ et al.) for GIB Dataset (time samples = 128).
    Supports 3D and 4D inputs.
    """
    def __init__(self, num_channels=8, time_samples=128):
        super().__init__()
        # Block 1: Conv2D filters=8, kernel=(8,15), max_norm=1.0, no bias
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(num_channels, 15), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout2d(0.35)
        
        # Block 2: Conv2D filters=16, kernel=(1,5), no bias
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout2d(0.20)
        
        # Block 3: Conv2D filters=32, kernel=(1,5), no bias
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d((1, 2))
        self.drop3 = nn.Dropout2d(0.40)
        
        # Automatically calculate flattening dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, time_samples)
            x = self.drop3(self.pool3(F.elu(self.bn3(self.conv3(self.drop2(self.pool2(F.elu(self.bn2(self.conv2(self.drop1(self.pool1(F.elu(self.bn1(self.conv1(dummy)))))))))))))))
            flat_size = x.view(1, -1).shape[1]
        
        # Dense classification layer: 2 classes, softmax, max_norm=0.25
        self.fc = nn.Linear(flat_size, 2, bias=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.drop1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        return self.softmax(self.fc(x.view(x.size(0), -1)))
