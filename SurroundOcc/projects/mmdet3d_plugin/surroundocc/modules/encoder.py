
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from SurroundOcc.projects.mmdet3d_plugin.models.utils.bricks import run_time
from SurroundOcc.projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
import pdb
import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccEncoder(TransformerLayerSequence):



    def __init__(self, *args, pc_range=None, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(OccEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z, bs=1, device='cuda', dtype=torch.float):


        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)
        return ref_3d



    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        volume_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        volume_mask = (volume_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            volume_mask = torch.nan_to_num(volume_mask)
        else:
            volume_mask = volume_mask.new_tensor(
                np.nan_to_num(volume_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, volume_mask

    @auto_fp16()
    def forward(self,
                volume_query,
                key,
                value,
                *args,
                volume_h=None,
                volume_w=None,
                volume_z=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):


        output = volume_query
        intermediate = []

        ref_3d = self.get_reference_points(
                    volume_h, volume_w, volume_z, bs=volume_query.size(1),  device=volume_query.device, dtype=volume_query.dtype)

        reference_points_cam, volume_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])



        volume_query = volume_query.permute(1, 0, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                volume_query,
                key,
                value,
                *args,
                ref_3d=ref_3d,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=volume_mask,
                **kwargs)

            volume_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class OccLayer(MyCustomBaseTransformerLayer):


    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 embed_dims,
                 ffn_dropout=0.0,
                 operation_order=None,
                 conv_num=1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(OccLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            embed_dims=embed_dims,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False

        self.deblock = nn.ModuleList()
        conv_cfg=dict(type='Conv3d', bias=False)
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        for i in range(conv_num):
            conv_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            deblock = nn.Sequential(conv_layer,
                                    build_norm_layer(norm_cfg, embed_dims)[1],
                                    nn.ReLU(inplace=True))
            self.deblock.append(deblock)




    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_3d=None,
                volume_h=None,
                volume_w=None,
                volume_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):


        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:

            if layer == 'conv':
                bs = query.shape[0]
                identity = query
                query = query.reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)
                for i in range(len(self.deblock)):
                    query = self.deblock[i](query)
                query = query.permute(0, 4, 3, 2, 1).reshape(bs, volume_z*volume_h*volume_w, -1)
                query = query + identity

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1


            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                if isinstance(query, tuple) or isinstance(query, list):
                    query = query[0]
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1


        return query
