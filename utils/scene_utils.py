import torch
from collections import defaultdict


def get_scene_list(data_loader):
    if isinstance(data_loader.dataset, torch.utils.data.Subset):
        original_dataset = data_loader.dataset.dataset
        data_infos = original_dataset.data_infos
        indices = data_loader.dataset.indices
    else:
        original_dataset = data_loader.dataset
        data_infos = original_dataset.data_infos
        indices = [i for i in range(len(data_infos))]
    scene_list = []
    scene_idxs_map = defaultdict(list)
    for i in indices:
        info = data_infos[i]
        scene_token = info['scene_token']
        if scene_token not in scene_list:
            scene_list.append(scene_token)
        scene_idxs_map[scene_token].append(i)

    start_idx_map = {}
    end_idx_map = {}
    i = 0
    for scene in scene_list:
        idxs = scene_idxs_map[scene]
        start_idx = idxs[0]
        end_idx = idxs[-1]
        start_idx_map[start_idx] = {'scene': str(scene), 'count': len(idxs), 'idx': i}
        end_idx_map[end_idx] = {'scene': str(scene), 'count': len(idxs), 'idx': i}
        i += 1
    return scene_list, scene_idxs_map, original_dataset, start_idx_map, end_idx_map
