
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import random
import os


@PIPELINES.register_module()
class LoadOccupancy(object):


    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic


    def __call__(self, results):
        occ = np.load(results['occ_path'])
        occ = occ.astype(np.float32)


        if self.use_semantic:
            occ[..., 3][occ[..., 3] == 0] = 255
        else:
            occ = occ[occ[..., 3] > 0]
            occ[..., 3] = 1

        results['gt_occ'] = occ


        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        return repr_str

