
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):


    def __call__(self, results):


        results = super(CustomDefaultFormatBundle3D, self).__call__(results)

        if 'gt_occ' in results.keys():
            results['gt_occ'] = DC(to_tensor(results['gt_occ']), stack=False)

        return results

