"""
BSKAN (Boundary-Selective KAN) module.
Replaces the standard MLP in Transformer blocks with a boundary-aware
KAN that selectively applies B-spline modeling to high-frequency regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    KAN linear layer using B-spline basis functions
    for learnable activation functions.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=None,
        medical_mode=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.medical_mode = medical_mode

        if grid_range is None:
            if medical_mode:
                grid_range = [-0.8, 0.8]
            else:
                grid_range = [-1, 1]
        self.grid_range = grid_range

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        if self.medical_mode:
            init_scale = math.sqrt(5) * self.scale_base * 0.8
        else:
            init_scale = math.sqrt(5) * self.scale_base

        nn.init.kaiming_uniform_(self.base_weight, a=init_scale)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                spline_scale = math.sqrt(5) * self.scale_spline
                if self.medical_mode:
                    spline_scale *= 0.8
                nn.init.kaiming_uniform_(self.spline_scaler, a=spline_scale)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.view(*original_shape[:-1], self.out_features)
        return output


class SobelBoundaryDetector(nn.Module):
    """
    Sobel-based boundary detector operating on token sequences.
    Produces per-token boundary scores via gradient magnitude estimation.
    """
    def __init__(self, in_features, hidden_ratio=4, refine_dim=8):
        super().__init__()
        self.in_features = in_features

        self.register_buffer('sobel_x', self._create_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._create_sobel_kernel('y'))

        self.channel_reduce = nn.Linear(in_features, min(64, in_features))
        self.reduced_channels = min(64, in_features)

        self.refine = nn.Sequential(
            nn.Linear(1, refine_dim),
            nn.ReLU(inplace=True),
            nn.Linear(refine_dim, 1),
            nn.Sigmoid()
        )

        self.temperature = nn.Parameter(torch.ones(1))

    def _create_sobel_kernel(self, direction):
        if direction == 'x':
            kernel = torch.tensor([
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ], dtype=torch.float32)
        else:
            kernel = torch.tensor([
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ], dtype=torch.float32)
        return kernel.view(1, 1, 3, 3)

    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N ** 0.5)

        if H * W != N:
            return self._mlp_fallback(x)

        x_reduced = self.channel_reduce(x)
        x_2d = x_reduced.transpose(1, 2).view(B, self.reduced_channels, H, W)

        sobel_x = self.sobel_x.expand(self.reduced_channels, 1, 3, 3)
        sobel_y = self.sobel_y.expand(self.reduced_channels, 1, 3, 3)

        x_padded = F.pad(x_2d, (1, 1, 1, 1), mode='reflect')
        g_x = F.conv2d(x_padded, sobel_x, groups=self.reduced_channels)
        g_y = F.conv2d(x_padded, sobel_y, groups=self.reduced_channels)

        gradient_magnitude = torch.sqrt(g_x ** 2 + g_y ** 2 + 1e-8)
        boundary_map = gradient_magnitude.mean(dim=1, keepdim=True)
        boundary_map = torch.sigmoid(boundary_map / (self.temperature.abs() + 1e-6))
        boundary_score = boundary_map.view(B, 1, H * W).transpose(1, 2)
        boundary_score = self.refine(boundary_score)

        return boundary_score

    def _mlp_fallback(self, x):
        x_shifted = torch.roll(x, shifts=1, dims=1)
        diff = torch.abs(x - x_shifted).mean(dim=-1, keepdim=True)
        return torch.sigmoid(diff)


class BoundaryAwareKANLinear(nn.Module):
    """
    Boundary-aware KAN linear layer.
    Routes boundary tokens through KAN and smooth tokens through MLP.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=3,
        spline_order=2,
        boundary_threshold=0.3,
        detector_hidden_ratio=4,
        dropout=0.1,
        temperature_init=1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.boundary_threshold = boundary_threshold

        self.boundary_detector = SobelBoundaryDetector(in_features, detector_hidden_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU()
        )

        self.kan = KANLinear(
            in_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_base=1.0,
            scale_spline=1.0,
            medical_mode=True
        )

        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

    def forward(self, x):
        boundary_score = self.boundary_detector(x)
        boundary_score = torch.sigmoid(
            (boundary_score - self.boundary_threshold) / self.temperature.abs()
        )

        mlp_out = self.mlp(x)
        kan_out = self.kan(x)

        output = boundary_score * kan_out + (1 - boundary_score) * mlp_out
        output = self.dropout(output)

        return output


class BSKAN_MLP(nn.Module):
    """
    Boundary-Selective KAN-MLP module.
    Drop-in replacement for the standard MLP in Transformer blocks.
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        grid_size=3,
        spline_order=2,
        boundary_threshold=0.3,
        dropout=0.1,
        temperature_init=1.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.boundary_kan = BoundaryAwareKANLinear(
            in_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            boundary_threshold=boundary_threshold,
            dropout=dropout,
            temperature_init=temperature_init
        )

    def forward(self, x):
        return self.boundary_kan(x)


class StandardMLP(nn.Module):
    """Standard MLP used in vanilla Transformer blocks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PureKAN_MLP(nn.Module):
    """Pure KAN MLP without boundary-aware routing."""
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        grid_size=3,
        spline_order=3,
        dropout=0.1
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.kan = KANLinear(
            in_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            medical_mode=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.kan(x)
        x = self.dropout(x)
        return x


def get_mlp_module(mlp_type, in_features, hidden_features=None, out_features=None,
                   grid_size=3, spline_order=2, boundary_threshold=0.3, dropout=0.1,
                   temperature_init=1.0):
    if mlp_type == 'mlp':
        return StandardMLP(in_features, hidden_features, out_features, dropout)
    elif mlp_type == 'kan':
        return PureKAN_MLP(in_features, hidden_features, out_features, grid_size,
                          spline_order, dropout)
    elif mlp_type == 'bskan':
        return BSKAN_MLP(in_features, hidden_features, out_features, grid_size,
                        spline_order, boundary_threshold, dropout, temperature_init)
    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")
