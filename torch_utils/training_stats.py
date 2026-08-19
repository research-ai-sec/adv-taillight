# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import re
import numpy as np
import torch
import dnnlib

from . import misc



_num_moments    = 3
_reduce_dtype   = torch.float32
_counter_dtype  = torch.float64
_rank           = 0
_sync_device    = None
_sync_called    = False
_counters       = dict()
_cumulative     = dict()



def init_multiprocessing(rank, sync_device):

    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device



@misc.profiled_function
def report(name, value):

    if name not in _counters:
        _counters[name] = dict()

    elems = torch.as_tensor(value)
    if elems.numel() == 0:
        return value

    elems = elems.detach().flatten().to(_reduce_dtype)
    moments = torch.stack([
        torch.ones_like(elems).sum(),
        elems.sum(),
        elems.square().sum(),
    ])
    assert moments.ndim == 1 and moments.shape[0] == _num_moments
    moments = moments.to(_counter_dtype)

    device = moments.device
    if device not in _counters[name]:
        _counters[name][device] = torch.zeros_like(moments)
    _counters[name][device].add_(moments)
    return value



def report0(name, value):

    report(name, value if _rank == 0 else [])
    return value



class Collector:

    def __init__(self, regex='.*', keep_previous=True):
        self._regex = re.compile(regex)
        self._keep_previous = keep_previous
        self._cumulative = dict()
        self._moments = dict()
        self.update()
        self._moments.clear()

    def names(self):

        return [name for name in _counters if self._regex.fullmatch(name)]

    def update(self):

        if not self._keep_previous:
            self._moments.clear()
        for name, cumulative in _sync(self.names()):
            if name not in self._cumulative:
                self._cumulative[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
            delta = cumulative - self._cumulative[name]
            self._cumulative[name].copy_(cumulative)
            if float(delta[0]) != 0:
                self._moments[name] = delta

    def _get_delta(self, name):

        assert self._regex.fullmatch(name)
        if name not in self._moments:
            self._moments[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
        return self._moments[name]

    def num(self, name):

        delta = self._get_delta(name)
        return int(delta[0])

    def mean(self, name):

        delta = self._get_delta(name)
        if int(delta[0]) == 0:
            return float('nan')
        return float(delta[1] / delta[0])

    def std(self, name):

        delta = self._get_delta(name)
        if int(delta[0]) == 0 or not np.isfinite(float(delta[1])):
            return float('nan')
        if int(delta[0]) == 1:
            return float(0)
        mean = float(delta[1] / delta[0])
        raw_var = float(delta[2] / delta[0])
        return np.sqrt(max(raw_var - np.square(mean), 0))

    def as_dict(self):

        stats = dnnlib.EasyDict()
        for name in self.names():
            stats[name] = dnnlib.EasyDict(num=self.num(name), mean=self.mean(name), std=self.std(name))
        return stats

    def __getitem__(self, name):

        return self.mean(name)



def _sync(names):

    if len(names) == 0:
        return []
    global _sync_called
    _sync_called = True


    deltas = []
    device = _sync_device if _sync_device is not None else torch.device('cpu')
    for name in names:
        delta = torch.zeros([_num_moments], dtype=_counter_dtype, device=device)
        for counter in _counters[name].values():
            delta.add_(counter.to(device))
            counter.copy_(torch.zeros_like(counter))
        deltas.append(delta)
    deltas = torch.stack(deltas)


    if _sync_device is not None:
        torch.distributed.all_reduce(deltas)


    deltas = deltas.cpu()
    for idx, name in enumerate(names):
        if name not in _cumulative:
            _cumulative[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
        _cumulative[name].add_(deltas[idx])


    return [(name, _cumulative[name]) for name in names]


