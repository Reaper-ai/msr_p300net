import torch
import torch.nn as nn
import torch.nn.functional as F

class MSR_P300NET(nn.Module):
    def __init__(self, n_channels=8, n_samples=128):
        super().__init__()
        self.c15 = nn.Conv2d(1, 4, (1, 15), padding=(0, 7), bias=False)
        self.c31 = nn.Conv2d(1, 4, (1, 31), padding=(0, 15), bias=False)
        self.c63 = nn.Conv2d(1, 4, (1, 63), padding=(0, 31), bias=False)
        self.bn_t = nn.BatchNorm2d(12)
        self.conv_s = nn.Conv2d(12, 24, (n_channels, 1), groups=12, bias=False)
        self.bn_s = nn.BatchNorm2d(24)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)

        self.c1_dw = nn.Conv2d(24, 24, (1, 5), padding=(0, 4), dilation=(1, 2), groups=24, bias=False)
        self.c1_pw = nn.Conv2d(24, 24, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.c2_dw = nn.Conv2d(24, 24, (1, 5), padding=(0, 4), dilation=(1, 2), groups=24, bias=False)
        self.c2_pw = nn.Conv2d(24, 24, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.25)

        self.c_dw = nn.Conv2d(24, 24, (1, 5), padding=(0, 2), groups=24, bias=False)
        self.c_pw = nn.Conv2d(24, 32, 1, bias=False)
        self.skip = nn.Conv2d(24, 32, 1, bias=False)
        self.bn_f = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d((1, 2))
        self.drop3 = nn.Dropout(0.30)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.drop1(self.pool1(F.elu(self.bn_s(self.conv_s(self.bn_t(torch.cat([self.c15(dummy), self.c31(dummy), self.c63(dummy)], 1)))))))
            x = self.drop2(self.pool2(F.elu(self.bn2(self.c2_pw(self.c2_dw(F.elu(self.bn1(self.c1_pw(self.c1_dw(x))))))) + x)))
            x = self.drop3(self.pool3(F.elu(self.bn_f(self.c_pw(self.c_dw(x))) + self.skip(x))))
            self.flat_dim = x.numel()

        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(self.flat_dim, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x1 = F.elu(self.bn_s(self.conv_s(self.bn_t(torch.cat([self.c15(x), self.c31(x), self.c63(x)], 1)))))
        x1 = self.drop1(self.pool1(x1))
        x2 = F.elu(self.bn1(self.c1_pw(self.c1_dw(x1))))
        x2 = F.elu(self.bn2(self.c2_pw(self.c2_dw(x2))) + x1)
        x2 = self.drop2(self.pool2(x2))
        x3 = F.elu(self.bn_f(self.c_pw(self.c_dw(x2))) + self.skip(x2))
        x3 = self.drop3(self.pool3(x3))
        return self.classifier(torch.flatten(x3, 1))

def apply_max_norm(model):
    with torch.no_grad():
        w = model.conv_s.weight
        nd1 = w.norm(2, dim=(1, 2, 3), keepdim=True)
        w.mul_(torch.clamp(nd1, max=1.0) / (nd1 + 1e-8))
        
        wd = model.classifier[-1].weight
        nd = wd.norm(2, dim=1, keepdim=True)
        wd.mul_(torch.clamp(nd, max=0.25) / (nd + 1e-8))
