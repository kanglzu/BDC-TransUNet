"""
BDC-TransUNet: Boundary-aware Deeply-supervised Converse Transformer
for Medical Image Segmentation.

Architecture:
    Encoder: ResNet50 + PriKAN-Former (BSKAN)
    Decoder: Converse2D upsampling + DSDA attention + deep supervision
"""

import copy
import logging
import math
from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

from .vit_seg_configs import CONFIGS
from .bskan_module import get_mlp_module, StandardMLP
from .converse_module import ConverseUpsample
from .dsda_module import DSDAHead

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": lambda x: x * torch.sigmoid(x)}




class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])
        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])
        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))
        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))
        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width))
                 for i in range(2, block_units[0] + 1)]))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2))
                 for i in range(2, block_units[1] + 1)]))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4))
                 for i in range(2, block_units[2] + 1)]))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert 0 < pad < 3, f"x {x.size()} should {right_size}"
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class Attention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Block(nn.Module):
    """Transformer block with pluggable MLP (standard / KAN / BSKAN)."""
    def __init__(self, config, vis, mlp_type='bskan', grid_size=3, spline_order=2, boundary_threshold=0.3):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis)

        self.mlp_type = mlp_type
        if mlp_type == 'mlp':
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.transformer["mlp_dim"]),
                nn.GELU(),
                nn.Dropout(config.transformer["dropout_rate"]),
                nn.Linear(config.transformer["mlp_dim"], config.hidden_size),
                nn.Dropout(config.transformer["dropout_rate"])
            )
        else:
            self.ffn = get_mlp_module(
                mlp_type,
                config.hidden_size,
                config.hidden_size,
                config.hidden_size,
                grid_size=grid_size,
                spline_order=spline_order,
                boundary_threshold=boundary_threshold,
                dropout=config.transformer["dropout_rate"]
            )

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block, skip_mlp=False):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            if not skip_mlp and self.mlp_type == 'mlp':
                mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
                mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
                mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
                mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

                self.ffn[0].weight.copy_(mlp_weight_0)
                self.ffn[3].weight.copy_(mlp_weight_1)
                self.ffn[0].bias.copy_(mlp_bias_0)
                self.ffn[3].bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, mlp_type='bskan', grid_size=3, spline_order=2, boundary_threshold=0.3):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, mlp_type, grid_size, spline_order, boundary_threshold)
            self.layer.append(layer)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = None
        self.config = config
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = config.patches["size"]
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor
            )
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, mlp_type='bskan', grid_size=3, spline_order=2, boundary_threshold=0.3):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, mlp_type, grid_size, spline_order, boundary_threshold)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features




class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super().__init__(conv, bn, relu)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (Hu et al., CVPR 2018)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0,
                 attention_type='dsda', upsample_type='converse',
                 use_batchnorm=True, da_reduction=16, num_classes=2):
        super().__init__()
        self.skip_channels = skip_channels
        self.attention_type = attention_type
        self.has_skip = skip_channels > 0

        # Upsampling
        if upsample_type == 'converse':
            self.up = ConverseUpsample(in_channels, in_channels, scale=2, mode='residual')
        elif upsample_type == 'deconv':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # Attention module
        concat_channels = skip_channels + in_channels if self.has_skip else in_channels
        if attention_type == 'dsda':
            self.dsda = DSDAHead(concat_channels, concat_channels,
                                num_classes=num_classes, reduction_ratio=da_reduction)
        elif attention_type == 'se':
            self.se = SEBlock(concat_channels, reduction=da_reduction)

        # Channel adjustment and refinement convolutions
        if self.has_skip:
            self.channel_adjust = Conv2dReLU(
                skip_channels + in_channels, out_channels,
                kernel_size=3, padding=1, use_batchnorm=use_batchnorm
            )
        else:
            self.channel_adjust = Conv2dReLU(
                in_channels, out_channels,
                kernel_size=3, padding=1, use_batchnorm=use_batchnorm
            )
        self.final_conv = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x, skip=None):
        x = self.up(x)

        aux_pred = None

        if skip is not None and self.has_skip:
            x = torch.cat([skip, x], dim=1)

        if self.attention_type == 'dsda':
            result = self.dsda(x)
            if isinstance(result, tuple):
                x, aux_pred = result
            else:
                x = result
        elif self.attention_type == 'se':
            x = self.se(x)

        x = self.channel_adjust(x)
        x = self.final_conv(x)

        if aux_pred is not None:
            return x, aux_pred
        return x


class Decoder(nn.Module):
    """Multi-scale decoder with configurable attention and upsampling."""
    def __init__(self, config, attention_type='dsda', upsample_type='converse',
                 enhance_layers=None, da_reduction=16, num_classes=2):
        super().__init__()
        self.config = config
        head_channels = 512

        if enhance_layers is None:
            enhance_layers = [0, 1, 2]
        self.enhance_layers = enhance_layers

        self.conv_more = Conv2dReLU(
            config.hidden_size, head_channels,
            kernel_size=3, padding=1, use_batchnorm=True
        )

        # Skip connection channels
        if self.config.n_skip != 0:
            skip_channels = list(self.config.skip_channels)
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        blocks = []
        for i, (in_ch, out_ch, sk_ch) in enumerate(zip(in_channels, out_channels, skip_channels)):
            layer_attn = attention_type if (i in enhance_layers) else 'none'
            blocks.append(DecoderBlock(
                in_ch, out_ch, sk_ch,
                attention_type=layer_attn,
                upsample_type=upsample_type,
                da_reduction=da_reduction,
                num_classes=num_classes
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        x = self.conv_more(x)

        aux_outputs = []
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            result = decoder_block(x, skip=skip)
            if isinstance(result, tuple):
                x, aux_pred = result
                aux_outputs.append(aux_pred)
            else:
                x = result

        if aux_outputs:
            return x, aux_outputs
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)





class BDCTransUNet(nn.Module):
    """
    BDC-TransUNet: Boundary-aware Deeply-supervised Converse Transformer.

    Args:
        config: ViT configuration dict
        img_size: input image resolution
        num_classes: number of segmentation classes
        encoder_type: encoder MLP variant ('bskan' default)
        attention_type: decoder attention ('dsda' default, or 'se', 'none')
        upsample_type: decoder upsampling ('converse' default, or 'bilinear', 'deconv')
        da_reduction: DSDA / SE channel reduction ratio
        grid_size: KAN/BSKAN grid size
        spline_order: KAN/BSKAN spline order
        boundary_threshold: BSKAN boundary detection threshold
        vis: whether to return attention weights
    """
    def __init__(
        self,
        config,
        img_size=224,
        num_classes=2,
        encoder_type='bskan',
        attention_type='dsda',
        upsample_type='converse',
        enhance_layers=None,
        da_reduction=16,
        grid_size=3,
        spline_order=2,
        boundary_threshold=0.3,
        vis=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.config = config
        self.encoder_type = encoder_type
        self.attention_type = attention_type
        self.upsample_type = upsample_type

        self.transformer = Transformer(
            config, img_size, vis,
            mlp_type=encoder_type,
            grid_size=grid_size,
            spline_order=spline_order,
            boundary_threshold=boundary_threshold
        )

        self.decoder = Decoder(
            config,
            attention_type=attention_type,
            upsample_type=upsample_type,
            enhance_layers=enhance_layers,
            da_reduction=da_reduction,
            num_classes=num_classes
        )

        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        decoder_out = self.decoder(x, features)

        if isinstance(decoder_out, tuple):
            x, aux_outputs = decoder_out
        else:
            x = decoder_out
            aux_outputs = None

        logits = self.segmentation_head(x)

        if aux_outputs is not None and self.training:
            return logits, aux_outputs
        return logits

    def load_from(self, weights, skip_mlp=None):
        """Load pretrained ViT + ResNet weights."""
        if skip_mlp is None:
            skip_mlp = self.encoder_type != 'mlp'

        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                np2th(weights["embedding/bias"])
            )

            self.transformer.encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings

            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname, skip_mlp=skip_mlp)

            if self.transformer.embeddings.hybrid:
                res_weight = weights
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True)
                )
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
