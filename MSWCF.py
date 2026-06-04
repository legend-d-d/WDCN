import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT1D(nn.Module):
    def __init__(self):
        super().__init__()
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        h = torch.tensor([1.0, 1.0]) * sqrt2_inv   
        g = torch.tensor([1.0, -1.0]) * sqrt2_inv  
        self.register_buffer("_h", h.view(1, 1, 2)) 
        self.register_buffer("_g", g.view(1, 1, 2))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, l = x.shape
        pad_right = (l % 2)
        if pad_right:
            x = F.pad(x, (0, 1), mode="replicate")
            l = l + 1
        h = self._h.repeat(c, 1, 1)
        g = self._g.repeat(c, 1, 1) 
        xl = F.conv1d(x, h, bias=None, stride=2, padding=0, groups=c)
        xh = F.conv1d(x, g, bias=None, stride=2, padding=0, groups=c)
        return xl, xh

class LearnableIWT1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.deconv = nn.ConvTranspose1d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=channels,
            bias=False,
        )
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        h = torch.tensor([1.0, 1.0]) * sqrt2_inv
        g = torch.tensor([1.0, -1.0]) * sqrt2_inv
        with torch.no_grad():
            w = torch.zeros((2 * channels, 1, 2), dtype=self.deconv.weight.dtype)
            for i in range(channels):
                w[2 * i + 0, 0, :] = h  
                w[2 * i + 1, 0, :] = g 
            self.deconv.weight.copy_(w)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.deconv(y)

class PW_Conv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        #padding = kernel_size // 2
        self.pw = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(x)
        
class MSWCF_module(nn.Module):   
    def __init__(self, channels: int, levels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.levels = levels
        padding = kernel_size // 2
        self.base_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.dwt = HaarDWT1D()        
        self.scale_convs = nn.ModuleList([
            PWConv1d(channels=2 * channels, kernel_size=kernel_size)
            for _ in range(levels)
        ])
        self.k1 = nn.ParameterList([
            nn.Parameter(torch.full((1, channels, 1), 1.0))
            for _ in range(levels)
        ])
        self.k2 = nn.ParameterList([
            nn.Parameter(torch.full((1, channels, 1), 1.0))
            for _ in range(levels)
        ])        
        self.iwt = LearnableIWT1D(channels=channels)
        self.conv1 = nn.Conv1d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, padding=kernel_size // 2, groups = channels, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, padding=kernel_size // 2, groups = channels, bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = x
        lows: List[torch.Tensor] = []
        highs: List[torch.Tensor] = []
        tent_res: List[torch.Tensor] = []
        x_low = x
        for i in range(self.levels):
            if i == 1:
                x_low = self.conv1(x_low)
            elif i == 2:
                x_low = self.conv2(x_low)
            tent_res.append(x_low) 
            xl, xh = self.dwt(x_low) 
            lows.append(xl)
            highs.append(xh)
            x_low = xl
 
        zhat_lows: List[torch.Tensor] = []
        zhat_highs: List[torch.Tensor] = []
        for i in range(self.levels):
            xbar = torch.cat([lows[i], highs[i]], dim=1)
            z = self.scale_convs[i](xbar)
            zhat = z
            zhat_lows.append(zhat[:, :self.channels, :])
            zhat_highs.append(zhat[:, self.channels:, :])

        r = torch.zeros_like(zhat_lows[-1])  # r_{L+1}=0
        for i in reversed(range(self.levels)): 
            Li = zhat_lows[i].shape[-1]
            if r.shape[-1] != Li:
                r = r[..., :Li]
            low_in = (zhat_lows[i]*self.k1[i] + r*self.k2[i])/(self.k1[i]+self.k2[i]+ 1e-8)
            high_in = zhat_highs[i]
            y = self.iwt(torch.cat([low_in, high_in], dim=1))
            r = y
        y_wave = r
        if y_wave.shape[-1] != l0:
            y_wave = y_wave[..., :l0]
        out = y_base + y_wave
        return out

class MSWCF_Block(nn.Module):
    def __init__(self, channels: int, levels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.mswcf = MSWCF_module(channels=channels, levels=levels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.mswcf(out)
        out = self.bn2(out)
        out = out + identity
        return out

class MSWCF_Block2(nn.Module):
    def __init__(self, channels: int, levels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.mswcf = MSWCF_module(channels=channels, levels=levels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(channels)
        self.shortcut = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.mswcf(out)
        out = self.bn2(out)
        out = out + identity
        return out

class MSWCF_Classifier(nn.Module):
    def __init__(self,
             in_channels: int = 1,
             channels: int = 32,
             levels: int = 3,
             kernel_size: int = 3,
             num_classes: int = 7
             ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        from my_modules import SincConv
        self.sinc_conv = SincConv(
            out_channels=32,    
            kernel_size=251,   
            sample_rate=64000,   
        )                    
        self.stem = nn.Sequential(
            DualDomainWaveletBlock(channels=channels, levels=levels, kernel_size=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        stages = []
        for _ in range(3):
            stages.append(MSWCF_Block(channels, levels, kernel_size=3))
            stages.append(MSWCF_Block2(channels, levels, kernel_size=3))
        self.stage3 = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.sinc_conv(x)
        out = self.stem(x) 
        out = self.stage3(out)
        out_1 = self.gap(out)
        out_1 = out_1.squeeze(-1)
        logits = self.fc(out_1)
        #return logits, out_1, out
        return logits, out_1





