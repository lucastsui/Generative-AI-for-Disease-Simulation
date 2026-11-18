from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 6
N_CLASSES = 3


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = N_CLASSES, base: int = 32, depth: int = 4):
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        c = in_ch
        for ch in chans:
            self.downs.append(ConvBlock(c, ch))
            c = ch
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for ch in reversed(chans[:-1]):
            self.ups.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.up_convs.append(ConvBlock(2 * ch, ch))
            c = ch
        self.head = nn.Conv2d(c, n_classes, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([x, skip], 1))
        return self.head(x)


def normalize(img01: torch.Tensor) -> torch.Tensor:
    valid = (img01 > 10 / 255).float()
    n = valid.sum((1, 2, 3), keepdim=True).clamp(min=1.0)
    mean = (img01 * valid).sum((1, 2, 3), keepdim=True) / n
    var = (((img01 - mean) ** 2) * valid).sum((1, 2, 3), keepdim=True) / n
    return (img01 - mean) / var.sqrt().clamp(min=0.02)


def pad_in(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (PAD, PAD, PAD, PAD))


def crop_out(x: torch.Tensor) -> torch.Tensor:
    return x[..., PAD:-PAD, PAD:-PAD]


@torch.no_grad()
def predict_probs(model: nn.Module, img_u8: torch.Tensor, tta: bool = True) -> torch.Tensor:
    model.eval()
    x = pad_in(normalize(img_u8.float().div(255).unsqueeze(1))).contiguous(memory_format=torch.channels_last)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
        logits = model(x).float()
        if tta:
            logits = logits + torch.flip(model(torch.flip(x, dims=[-1])).float(), dims=[-1])
            logits = logits / 2
    return crop_out(F.softmax(logits, 1))
