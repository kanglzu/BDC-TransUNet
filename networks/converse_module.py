"""
Converse Decoder upsampling modules.

Provides FFT-based inverse convolution upsampling (Converse2D)
and its residual variant (ResidualConverse2D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearUpsample(nn.Module):
    """Standard bilinear interpolation upsampling."""
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x


class TransposedConvUpsample(nn.Module):
    """Transposed convolution upsampling."""
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Converse2D_Base(nn.Module):
    """
    Base class for Converse2D.
    Implements the closed-form FFT solver for regularized inverse convolution.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        scale=1,
        padding=4,
        padding_mode='circular',
        eps=1e-5,
        lambda_init=0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps
        self.lambda_init = lambda_init

        assert out_channels == in_channels, "Converse2D requires out_channels == in_channels"

        self.weight = nn.Parameter(
            torch.randn(1, in_channels, kernel_size, kernel_size)
        )
        self.weight.data = F.softmax(
            self.weight.data.view(1, in_channels, -1), dim=-1
        ).view(1, in_channels, kernel_size, kernel_size)

        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def _converse_solve(self, x, lambda_reg):
        """Closed-form FFT solver."""
        _, _, h, w = x.shape

        STy = self._upsample_zeros(x, self.scale)

        if self.scale != 1:
            x0 = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        else:
            x0 = x

        FB = self._p2o(self.weight, (h * self.scale, w * self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)

        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
        FR = FBFy + torch.fft.fftn(lambda_reg * x0, dim=(-2, -1))

        x1 = FB * FR
        FBR = torch.mean(self._splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self._splits(F2B, self.scale), dim=-1, keepdim=False)

        if lambda_reg.dim() == 4 and lambda_reg.shape[-1] > 1 and self.scale > 1:
            lambda_reg_small = torch.mean(self._splits(lambda_reg, self.scale), dim=-1, keepdim=False)
        else:
            lambda_reg_small = lambda_reg

        invWBR = FBR / (invW + lambda_reg_small)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)

        FX = (FR - FCBinvWBR) / lambda_reg
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        return out

    def _splits(self, a, scale):
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale
        b = a.view(*leading_dims, scale, W_s, scale, H_s)
        permute_order = list(range(len(leading_dims))) + [
            len(leading_dims) + 1, len(leading_dims) + 3,
            len(leading_dims), len(leading_dims) + 2
        ]
        b = b.permute(*permute_order).contiguous()
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    def _p2o(self, psf, shape):
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2]/2), -int(psf.shape[-1]/2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2, -1))
        return otf

    def _upsample_zeros(self, x, scale):
        if scale == 1:
            return x
        z = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)
        ).type_as(x)
        z[..., ::scale, ::scale].copy_(x)
        return z


class Converse2D(Converse2D_Base):
    """Standard Converse2D upsampling."""
    def forward(self, x):
        if self.padding > 0:
            x_padded = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
        else:
            x_padded = x

        lambda_reg = torch.sigmoid(self.bias + self.lambda_init) + self.eps
        out = self._converse_solve(x_padded, lambda_reg)

        if self.padding > 0:
            p = self.padding * self.scale
            out = out[..., p:-p, p:-p]

        return out


class ResidualConverse2D(Converse2D_Base):
    """
    Residual Converse2D.
    output = bilinear_upsample(x) + alpha * converse_detail(x)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        scale=1,
        padding=4,
        padding_mode='circular',
        eps=1e-5,
        lambda_init=0.0,
        learnable_alpha=True,
        alpha_init=0.5
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, scale,
            padding, padding_mode, eps, lambda_init
        )

        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1) * alpha_init)
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init))

    def forward(self, x):
        if self.scale != 1:
            base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        else:
            base = x

        if self.padding > 0:
            x_padded = F.pad(x, [self.padding] * 4, mode=self.padding_mode)
        else:
            x_padded = x

        lambda_reg = torch.sigmoid(self.bias + self.lambda_init) + self.eps
        detail = self._converse_solve(x_padded, lambda_reg)

        if self.padding > 0:
            p = self.padding * self.scale
            detail = detail[..., p:-p, p:-p]

        alpha = torch.sigmoid(self.alpha) if self.learnable_alpha else self.alpha
        out = base + alpha * (detail - base)

        return out


class ConverseUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale=2,
        kernel_size=5,
        lambda_init=0.0,
        mode='residual',
        alpha_init=0.5
    ):
        super().__init__()

        self.mode = mode
        self.channel_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        padding = kernel_size - 1

        if mode == 'original':
            self.converse = Converse2D(
                out_channels, out_channels,
                kernel_size=kernel_size, scale=scale,
                padding=padding, lambda_init=lambda_init
            )
        elif mode == 'residual':
            self.converse = ResidualConverse2D(
                out_channels, out_channels,
                kernel_size=kernel_size, scale=scale,
                padding=padding, lambda_init=lambda_init,
                alpha_init=alpha_init
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.channel_conv(x)
        x = self.converse(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def get_upsample_module(upsample_type, in_channels, out_channels, scale=2):
    if upsample_type == 'bilinear':
        return BilinearUpsample(in_channels, out_channels, scale)
    elif upsample_type == 'transposed':
        return TransposedConvUpsample(in_channels, out_channels, scale)
    elif upsample_type == 'converse':
        return ConverseUpsample(in_channels, out_channels, scale, mode='original')
    elif upsample_type == 'converse_residual':
        return ConverseUpsample(in_channels, out_channels, scale, mode='residual')
    else:
        raise ValueError(f"Unknown upsample type: {upsample_type}")
