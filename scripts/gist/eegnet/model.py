import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNetv4(nn.Module):
    """
    Lawhern et al. 2018 EEGNet v4 architecture modified to output 1 logit.
    """
    def __init__(self, in_chans=8, n_classes=1, input_window_samples=128, F1=8, D=2, F2=16, kernel_length=64, drop_prob=0.25):
        super().__init__()
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (in_chans, 1), groups=F1, bias=False)
        self.bnorm_1 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.pool_1 = nn.AvgPool2d((1, 4))
        self.drop_1 = nn.Dropout(drop_prob)
        self.conv_separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv_separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bnorm_2 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.pool_2 = nn.AvgPool2d((1, 8))
        self.drop_2 = nn.Dropout(drop_prob)
        
        with torch.no_grad():
            dummy = torch.ones(1, 1, in_chans, input_window_samples)
            x = self.drop_1(self.pool_1(F.elu(self.bnorm_1(self.conv_spatial(self.bnorm_temporal(self.conv_temporal(dummy)))))))
            x = self.drop_2(self.pool_2(F.elu(self.bnorm_2(self.conv_separable_point(self.conv_separable_depth(x))))))
            self.flat_dim = x.numel()
        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def forward(self, x):
        if x.dim() == 3: 
            x = x.unsqueeze(1)
        x = F.elu(self.bnorm_temporal(self.conv_temporal(x)))
        x = F.elu(self.bnorm_1(self.conv_spatial(x)))
        x = self.drop_1(self.pool_1(x))
        x = F.elu(self.bnorm_2(self.conv_separable_point(self.conv_separable_depth(x))))
        x = self.drop_2(self.pool_2(x))
        return self.classifier(x.view(x.size(0), -1))

def apply_eegnet_constraints(model):
    with torch.no_grad():
        if hasattr(model, 'conv_spatial'):
            w = model.conv_spatial.weight
            norm = w.norm(2, dim=(2, 3), keepdim=True)
            w.mul_(torch.clamp(norm, max=1.0) / (norm + 1e-8))
        if hasattr(model, 'classifier'):
            w = model.classifier.weight
            norm = w.norm(2, dim=1, keepdim=True)
            w.mul_(torch.clamp(norm, max=0.25) / (norm + 1e-8))
