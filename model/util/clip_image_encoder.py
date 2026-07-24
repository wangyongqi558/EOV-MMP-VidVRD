import logging
import math

import torch
import torch.nn as nn
from clip import clip
from typing import Dict
from util.misc import NestedTensor
from collections import OrderedDict
import torch.nn.functional as F
from util.detectron2.modeling.poolers import ROIPooler
from util.custom_activation import MultiheadAttention
import cv2
import torchvision.transforms as T

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # when use offline modules, avoid overwriting running mean and var for loaded weights
            skip_reset = False
            for k_n in state_dict: # checkpoint weights
                if 'ignore_others' in k_n: #if 'offline' in k_n:
                    skip_reset = True
            if not skip_reset:
                # No running_mean/var in early versions
                # This will silent the warnings
                if prefix + "running_mean" not in state_dict:
                    state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
                if prefix + "running_var" not in state_dict:
                    state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        # NOTE: if a checkpoint is trained with BatchNorm and loaded (together with
        # version number) to FrozenBatchNorm, running_var will be wrong. One solution
        # is to remove the version number from the checkpoint.
        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, norm=nn.BatchNorm2d):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", norm(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class ResNetImageEncoder(nn.Module):

    def __init__(self, backbone_name, norm=FrozenBatchNorm2d):

        super(ResNetImageEncoder, self).__init__()

        clip_model, _ = clip.load(backbone_name, device="cpu")
        weights = clip_model.visual.float().state_dict()

        for name in list(weights.keys()):
            if "num_batches_tracked" in name:
                del weights[name]

        counts: list = [len(set(k.split(".")[1] for k in weights if k.startswith(f"layer{b}"))) for b in [1, 2, 3, 4]]
        layers = tuple(counts)
        width = weights["layer1.0.conv1.weight"].shape[0]

        if backbone_name == 'RN50':
            output_dim = 1024
            input_resolution = 224
        elif backbone_name == 'RN50x4':
            output_dim = 640
            input_resolution = 288
        elif backbone_name == 'RN50x16':
            output_dim = 768
            input_resolution = 384
        elif backbone_name == 'RN50x64':
            output_dim = 1024
            input_resolution = 448


        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = norm(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], norm=norm)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, norm=norm)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, norm=norm)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, norm=norm)

        embed_dim = width * 32  # the ResNet feature dimension
        visual_heads = width * 32 // 64
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, visual_heads, output_dim)

        self.load_state_dict(weights)

        # Frozen
        for p in self.parameters(): p.requires_grad = False

    def stem(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def _make_layer(self, planes, blocks, stride=1, norm=nn.BatchNorm2d):
        layers = [Bottleneck(self._inplanes, planes, stride, norm=norm)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, tensor_list: NestedTensor):

        x = tensor_list.tensors
        x = x.type(self.conv1.weight.dtype)
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

class RoIAlignImageEncoder(ResNetImageEncoder):

    def __init__(self, backbone_name,
                 spatial_scale=[1/16.0], sampling_ratio=0, output_size=14):

        super().__init__(backbone_name)

        self.roi_align = ROIPooler(
            output_size=output_size,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )

    def forward(self, tensor_list, roi_boxes):

        x = tensor_list.tensors
        x = x.type(self.conv1.weight.dtype)
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.roi_align([x3], roi_boxes)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head, attn_mask)
        self.n_head = n_head
        self.d_model = d_model
        self.scale = n_head ** -0.5

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, roi_indices=None, last=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, roi_indices=roi_indices, last=last)[0]

    def forward(self, x: torch.Tensor, roi_indices=None, last=False):

        if roi_indices is None:
            if not last:
                x = x + self.attention(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
            else:
                x = x + self.attention(self.ln_1(x), last=last)
                x = self.ln_2(x)
        else:
            _x = []
            _num_boxes = [len(rois) for rois in roi_indices]
            for idx in range(len(_num_boxes)):
                _x.append(x[0, idx, :].unsqueeze(0).expand(_num_boxes[idx], -1))
            _x = torch.cat(_x, dim=0)
            x = _x + self.attention(self.ln_1(x), roi_indices=roi_indices)
            x = self.ln_2(x) # x + self.mlp(self.ln_2(x))
        return x

class ViTImageEncoder(nn.Module):

    def __init__(self, clip_name):
        super(ViTImageEncoder, self).__init__()

        if clip_name == 'ViT-B/16':
            self.input_resolution = 224
            self.output_dim = 512
            self.width = 768
            self.patch_size = 16
            self.layers = 12
            self.heads = 12
        elif clip_name == 'ViT-B/32':
            self.input_resolution = 224
            self.output_dim = 512
            self.width = 768
            self.patch_size = 32
            self.layers = 12
            self.heads = 12
        elif clip_name == 'ViT-L/14':
            self.input_resolution = 224
            self.output_dim = 768
            self.width = 1024
            self.patch_size = 14
            self.layers = 24
            self.heads = 16
        elif clip_name == 'ViT-L/14@336px':
            self.input_resolution = 336
            self.output_dim = 768
            self.width = 1024
            self.patch_size = 14
            self.layers = 24
            self.heads = 16

        '''
        if clip_name == 'ViT-B/16':
            self.patch_size = 16
            self.orig_size = 224
        elif clip_name == 'ViT-B/32':
            self.patch_size = 32
            self.orig_size = 224
        elif clip_name == 'ViT-L/14':
            self.patch_size = 14
            self.orig_size = 224
        '''

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.width,
                               kernel_size=self.patch_size, stride=self.patch_size, bias=False)

        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.input_resolution // self.patch_size) ** 2 + 1, self.width))
        self.ln_pre = LayerNorm(self.width)

        self.transformer = Transformer(self.width, self.layers, self.heads)

        self.ln_post = LayerNorm(self.width)
        self.proj = nn.Parameter(scale * torch.randn(self.width, self.output_dim))

        clip_model, _ = clip.load(clip_name, device="cpu")
        weights = clip_model.visual.float().state_dict()
        self.load_state_dict(weights)

        # Frozen
        for p in self.parameters(): p.requires_grad = False

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):

        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        patch_pos_embed = pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.input_resolution // self.patch_size, self.input_resolution // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)

        return scale_pos_embed

    def forward(self, x: torch.Tensor):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos_embed = self.positional_embedding.to(x.dtype).unsqueeze(0)

        # interpolate init pe
        if (pos_embed.shape[1]) != x.shape[1]:
            pos_embed = self.InterpolateInitPosEmbed(pos_embed, img_size=(H, W))

        x = x + pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class RoIAlignViTImageEncoder(ViTImageEncoder):
    def __init__(self, backbone_name, last_stage_blk=11,
                 spatial_scale=[1/16.0], sampling_ratio=0, output_size=14):

        super().__init__(backbone_name)
        self.last_stage_blk = last_stage_blk

        self.roi_align = ROIPooler(
            output_size=output_size,
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
            canonical_box_size=224, #50 48
            pooler_type="ROIAlignV2",# ROIAlign ROIPool ROIAlignV2
        )
        self.resize = T.Resize((336, 336))
        self.clip_model, _ = clip.load(backbone_name, device="cpu")
        weights = self.clip_model.visual.float().state_dict()
        self.model = self.clip_model.visual
        self.model.load_state_dict(weights)
        self.model = self.model.cuda()

    '''
    def forward(self, tensor_list, roi_boxes):

        #x = tensor_list.tensors
        # x = tensor_list.tensors
        if isinstance(tensor_list, NestedTensor):
            x = tensor_list.tensors
        else:
            x = tensor_list
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        num_h_token = H // self.patch_size
        num_w_token = W // self.patch_size

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos_embed = self.positional_embedding.to(x.dtype).unsqueeze(0)

        # interpolate init pe
        if (pos_embed.shape[1]) != x.shape[1]:
            pos_embed = self.InterpolateInitPosEmbed(pos_embed, img_size=(H, W))

        x = x + pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2) # NLD -> LND

        # here we need to do something.
        #x = self.model.transformer(x)
        blocks = self.transformer.resblocks
        for blk in range(23):
            x = blocks[blk](x)

        x = x.permute(1, 0, 2)   # LND -> NLD
        cls, patches = x[:, 0, :], x[:, 1:, :].view(B, num_h_token, num_w_token, -1)

        # interpolation
        patches = patches.permute(0, 3, 1, 2) # [box, c, h, w]
        patches = F.interpolate(patches, [H//16, W//16], mode='bilinear')
        patches = self.roi_align([patches], roi_boxes)

        num_boxes, _c, _h, _w = patches.shape # [box, c, _h, _w]
        patches = patches.permute(0, 2, 3, 1).contiguous()
        patches = patches.view(num_boxes, _h * _w, -1)

        # cls token
        cls_tokens = []
        for idx, box in enumerate(roi_boxes):
            num = len(box.tensor)
            _cls = cls[idx].unsqueeze(0).unsqueeze(0).expand(num, -1, -1)
            cls_tokens.append(_cls)
        cls_tokens = torch.cat(cls_tokens, dim=0)

        x = torch.cat([cls_tokens, patches], dim=1)
        x = x.permute(1, 0, 2) # [C, B,

        # last stage
        for blk in range(23, 24):
            x = blocks[blk](x)
        #x = blocks[11](x, last=False)

        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj


        return x
    
    '''
    '''
    
    # Naiive algorithm
    def forward(self, tensor_list, roi_boxes):
        if isinstance(tensor_list, NestedTensor):
            x = tensor_list.tensors
        else:
            x = tensor_list

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        result = []
        for bs in range(B):
            x_clip = []
            box = roi_boxes[bs]
            for b in box.tensor:
                x1, y1, x2, y2 = b
                # print(x.shape)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                # if x2 - x1 < 16 or y2 - y1 < 16:
                #     continue

                x1, y1, x2, y2 = math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2)
                if x1 == x2:
                    x2 += 1
                if y1 == y2:
                    y2 += 1
                # print(x1, y1, x2, y2)
                tmp_x = x[bs, :, y1:y2, x1:x2].squeeze(0)

                tmp_x = self.resize(tmp_x)
                x_clip.append(tmp_x)
            x_clip = torch.stack(x_clip)
            # x_clip = x_clip.type(torch.HalfTensor)
            x_clip = x_clip.cuda()
            out = self.model(x_clip)
            # if self.proj is not None:
            #     out = out @ self.proj
            result.append(out)
        result = torch.cat(result)
        # result = result.squeeze(0)

        return result
    
    '''
    # Mask Attention
    def forward(self, tensor_list, roi_boxes):

        if isinstance(tensor_list, NestedTensor):
            x = tensor_list.tensors
        else:
            x = tensor_list

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        pos_embed = self.positional_embedding.to(x.dtype).unsqueeze(0)

        # interpolate init pe
        if (pos_embed.shape[1]) != x.shape[1]:
            pos_embed = self.InterpolateInitPosEmbed(pos_embed, img_size=(H, W))

        x = x + pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2) # NLD -> LND

        roi_indices = [boxes.tensor.to(torch.int32) // self.patch_size for boxes in roi_boxes]

        blocks = self.transformer.resblocks
        # print(f"length of blocks: {len(blocks)}")
        active_blk = 23
        for blk in range(len(blocks)):
            # print(f"{blk} : {x.shape}")
            if blk == active_blk:
                x = blocks[blk](x, roi_indices)
                break
            else:
                x = blocks[blk](x)

        # 38.0 22.5 /

        #
        #x1, x2 = x[0, :, :], x[1:, :, :]
        #HW, B, C = x2.shape
        #x2 = x2.view(H // self.patch_size, W // self.patch_size, B, C).permute(2, 3, 0, 1)
        #tmp = 52
        #x2 = F.interpolate(x2, [tmp, tmp], mode='bilinear')
        #x2 = x2.permute(2, 3, 0, 1).view(tmp*tmp, B, C)
        #x = torch.cat([x1.unsqueeze(0), x2], dim=0)
        #
        # x = blocks[-1](x, roi_indices)
        # print(f"After blks: {x.shape}")
        x = self.ln_post(x)
        # print(f"After LN POST: {x.shape}")
        if self.proj is not None:
            x = x @ self.proj

        return x


'''
torch.Size([4, 3, 1056, 1200])
torch.Size([4, 1024, 66, 75])

model = RoIAlignImageEncoder('ViT-B/16', patch_size=16)

input = torch.rand((1, 3, 420, 420), dtype=torch.float32)
output = model(input, roi_boxes=None)
print(output.shape)
'''