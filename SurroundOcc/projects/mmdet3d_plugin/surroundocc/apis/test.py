# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util

import pdb

def custom_encode_mask_results(mask_results):

    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, is_vis=False):


    model.eval()
    occ_results = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)

    for i, data in enumerate(data_loader):
        with torch.no_grad():

            result = model(return_loss=False, rescale=True, **data)

            if isinstance(result, dict):
                if 'evaluation' in result.keys():
                    occ_results.extend(result['evaluation'])
                    batch_size = len(result['evaluation'])

            if is_vis:
                batch_size = result

        if rank == 0:

            for _ in range(batch_size * world_size):
                prog_bar.update()

    if is_vis:
        return


    if gpu_collect:
        occ_results = collect_results_gpu(occ_results, len(dataset))
    else:
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        occ_results = collect_results_cpu(occ_results, len(dataset), tmpdir)

    return occ_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()

    if tmpdir is None:
        MAX_LEN = 512

        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)

    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank != 0:
        return None
    else:

        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))

        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''

        for res in part_list:
            ordered_results.extend(list(res))

        ordered_results = ordered_results[:size]

        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)

