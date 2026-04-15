"""
DSDA (Deeply-Supervised Dual Attention) module.

Combines Position Attention (PAM) and Channel Attention (CAM) with
auxiliary supervision heads at each decoder stage, enforcing
resolution-specific attention learning.

Reference:
    PAM/CAM: DANet (Fu et al., CVPR 2019)
    Deep Supervision: Lee et al., AISTATS 2015
"""

import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax


class PAM_Module(Module):
    """Position Attention Module (PAM)."""
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        reduced_dim = max(1, in_dim // 8)
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=reduced_dim, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=reduced_dim, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """Channel Attention Module (CAM)."""
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class DSDAHead(nn.Module):
    """Deeply-Supervised Dual Attention Head."""
    def __init__(self, in_channels, out_channels, num_classes=2, reduction_ratio=16, max_spatial=None):
        super().__init__()
        inter_channels = max(1, in_channels // reduction_ratio)
        self.max_spatial = max_spatial

        # Position attention branch
        self.conv5a = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.sa = PAM_Module(inter_channels)
        self.conv51 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        # Channel attention branch
        self.conv5c = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.sc = CAM_Module(inter_channels)
        self.conv52 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )

        # Aggregation output
        self.conv8 = nn.Sequential(
            nn.Dropout2d(0.05, False),
            nn.Conv2d(inter_channels, out_channels, 1),
            nn.ReLU()
        )

        # Auxiliary supervision head
        self.aux_head = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout2d(0.05, False),
            nn.Conv2d(inter_channels, num_classes, 1)
        )

    def forward(self, x):
        original_size = x.shape[2:]

        need_upsample = False
        if self.max_spatial is not None:
            H, W = original_size
            if H > self.max_spatial or W > self.max_spatial:
                scale = self.max_spatial / max(H, W)
                new_H, new_W = int(H * scale), int(W * scale)
                x = nn.functional.avg_pool2d(x, kernel_size=H // new_H, stride=H // new_H)
                need_upsample = True

        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        if need_upsample:
            feat_sum = nn.functional.interpolate(
                feat_sum, size=original_size, mode='bilinear', align_corners=False
            )

        sasc_output = self.conv8(feat_sum)

        if self.training:
            aux_logits = self.aux_head(feat_sum)
            return sasc_output, aux_logits

        return sasc_output
