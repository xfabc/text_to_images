import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, in_channels=131, out_channels=3, text_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 128)
        self.conv_in = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.resblock1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
        )
        self.upblock1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU()
        )
        self.upblock2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_out = nn.Conv2d(64, out_channels, 3, 1, 1)

    def forward(self, x, t, text_feat):
        # print("x",x.shape)
        # print("t",t.shape)
        t_emb = self.text_proj(text_feat).view(x.size(0), 1, 1, -1)
        t_emb = t_emb.expand(-1, x.size(2), x.size(3), -1).permute(0, 3, 1, 2)
        # print("t_emb",t_emb.shape)
        x = torch.cat([x, t_emb], dim=1)
        # print("conv_in",x.shape)
        x = self.conv_in(x)
        # print(x.shape)
        x1 = self.resblock1(x)
        x2 = self.resblock2(x1)
        x = self.upblock1(x2)
        x = self.upblock2(x + x1)
        return self.conv_out(x)