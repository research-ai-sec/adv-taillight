import torch
from detector.voxel_car_detector import detect_valid_car_in_range, detect_all_cars_with_ret, get_roi_mask, downsample_3d_any
from enum import Enum
import numpy as np

class LossType(Enum):
    softmax_reduce_other_sum_log = 'softmax_reduce_other_sum_log'
    softmax_empty_road = 'softmax_empty_road'
    softmax_empty_road_max = 'softmax_empty_road_max'
    logit_empty_road_max = 'logit_empty_road_max'
    logit_softplus_empty_road_max= 'logit_softplus_empty_road_max'
    logit_softplus_empty_road = 'logit_softplus_empty_road'
    softmax_softplus_empty_road = 'softmax_softplus_empty_road'
    softmax_softplus_empty_road_max = 'softmax_softplus_empty_road_max'

    logit_softplus_paper = 'logit_softplus_paper'


    energy_reallocation_logit = "energy_reallocation_logit"
    energy_reallocation_softmax = "energy_reallocation_softmax"
    entropy_reshaping_inner = "entropy_reshaping_inner"
    kl_softmax = "kl_softmax"
    cos_softmax = "cos_softmax"
    kl_logit = "kl_logit"
    cos_logit = "cos_logit"

class AttentionLossType(Enum):
    out_ref_point = "out_ref_point"
    attention = "attention"
    out_att = "out_att"
    out_att_adv = "out_att_adv"
    att_adv = "att_adv"
    adv = 'adv'
    point_in = 'point_in'
    pint_in_att_adv = 'pint_in_att_adv'


    adv_roi_point_enhance_att = 'adv_roi_point_enhance_att'
    roi_point_enhance_att = 'roi_point_enhance_att'
    adv_roi_point_supress_att = 'adv_roi_point_supress_att'
    adv_roi_supress_att = 'adv_roi_supress_att'
    adv_roi_point_in = 'adv_roi_point_in'
    roi_supress_att = 'roi_supress_att'
    roi_point_supress_att = 'roi_point_supress_att'

    adv_roi_point_in_patch = 'adv_roi_point_in_patch'
    adv_roi_point_enhance_att_patch = 'adv_roi_point_enhance_att_patch'


    diver = 'diver'
    adv_diver = 'adv_diver'


def trans_lidar_to_img(lidar2img_rt, points):

    if points.ndim == 1:
        points = points.reshape(1, 3)
    n = points.shape[0]
    P_ego_hom = np.column_stack([points, np.ones(n)])

    p = (lidar2img_rt @ P_ego_hom.T).T

    Z_c = p[:,2]
    if np.any(Z_c < 0.001):
        return None
    p_x = p[:,0] / Z_c
    p_y = p[:,1] / Z_c
    p = np.stack([p_x, p_y],axis=-1)

    return p


def get_target_points_torch(mask_tensor, num_points=10):


    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor[0]

    H, W = mask_tensor.shape


    coords = torch.nonzero(mask_tensor, as_tuple=False)  

    if coords.numel() == 0:

        return torch.empty((0, 2), dtype=torch.float32, device=mask_tensor.device)


    ys = coords[:, 0]
    xs = coords[:, 1]


    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()


    y_center = (y_min + y_max).to(torch.float32) / 2.0


    all_coords = torch.linspace(
        x_min.to(torch.float32),
        x_max.to(torch.float32),
        steps=num_points + 2,
        device=mask_tensor.device
    )
    x_coords = all_coords[1:-1]


    y_coords = torch.full_like(x_coords, y_center)


    points = torch.stack([x_coords, y_coords], dim=1)


    norm_scale = torch.tensor([W, H], dtype=torch.float32, device=points.device)
    points = points / norm_scale
    points = torch.clamp(points, 0.0, 1.0)

    points = points.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    return points


def sample_location_mask_in_roi(volume_mask, indexes, sample_location, volume_h, volume_w, volume_z):


    roi_mask = get_roi_mask().to(sample_location.device).to(sample_location.dtype)
    roi_mask_reshape = downsample_3d_any(roi_mask, target_size=(volume_w, volume_h,  volume_z)).to(sample_location.device).to(sample_location.dtype)
    roi_mask_reshape = roi_mask_reshape.permute(2,1,0)
    roi_mask_reshape = roi_mask_reshape.reshape(-1)


    n_cam, bs, n_points,D = volume_mask.shape
    num_cams, n_valid_points, num_heads, num_levels, num_all_points, xy = sample_location.shape

    indexes_from_volume_mask = []
    for i, mask_per_img in enumerate(volume_mask):
        index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
        indexes_from_volume_mask.append(index_query_per_img)

    max_len = max([len(each) for each in indexes])
    index_query_front =  indexes[0].to(sample_location.device)
    roi_mask_rebatch = torch.zeros((max_len), dtype=sample_location.dtype, device=sample_location.device)
    roi_mask_rebatch[:len(index_query_front)] = roi_mask_reshape[index_query_front]
    roi_mask_rebatch = roi_mask_rebatch.unsqueeze(1).expand(-1, num_heads*num_levels*num_all_points)
    roi_mask_rebatch = roi_mask_rebatch.reshape(-1)


    return roi_mask_rebatch


def create_point_mask_from_patch(patch_mask, normalized_points):

    if patch_mask.dim() == 3:
        patch_mask = patch_mask[0]
    elif patch_mask.dim() == 4:
        patch_mask = patch_mask[0][0]

    H, W = patch_mask.shape


    nonzero_indices = torch.nonzero(patch_mask, as_tuple=False)
    y_coords = nonzero_indices[:, 0].float() / float(H)
    x_coords = nonzero_indices[:, 1].float() / float(W)
    ymin = y_coords.min().item()
    ymax = y_coords.max().item()
    xmin = x_coords.min().item()
    xmax = x_coords.max().item()


    if normalized_points.dim() == 3 and normalized_points.size(1) == 1:
        pts = normalized_points.squeeze(1)
    else:
        pts = normalized_points


    eps = 1e-7
    pts_x = pts[:, 0]
    pts_y = pts[:, 1]
    in_x = (pts_x >= xmin) & (pts_x <= xmax)
    in_y = (pts_y >= ymin) & (pts_y <= ymax)
    point_mask = (in_x & in_y).to(normalized_points.dtype)

    return point_mask


def roi_point_in_loss_func(patch_mask, sampling_locations_list, volum_size_list,volume_mask_list, indexes_list, only_out_patch=False):
    if sampling_locations_list is None or len(sampling_locations_list) == 0:
        return 0
    loss = 0
    layer_count = 0
    for i, sample_location in enumerate(sampling_locations_list):
        if i in [0,1,4]:
            continue
        volume_h, volume_w, volume_z = volum_size_list[i]
        volume_mask = volume_mask_list[i]
        indexes = indexes_list[i]

        target_ref_points_cam = get_target_points_torch(patch_mask)
        target_ref_points_cam = target_ref_points_cam.reshape(1, -1,2)
        location_in_roi = sample_location_mask_in_roi(volume_mask, indexes, sample_location, volume_h, volume_w, volume_z)
        sample_location_front = sample_location[0]
        sample_location_front_reshape = sample_location_front.reshape(-1,1,2)
        diff = sample_location_front_reshape - target_ref_points_cam

        distance = torch.norm(diff, p=2, dim=-1)
        min_diff,_ = torch.min(distance, dim=1)
        min_diff_in_roi = min_diff * location_in_roi

        if only_out_patch:
            is_inside_mask = create_point_mask_from_patch(patch_mask, sample_location_front_reshape)
            penalty_mask = 1.0 - is_inside_mask
            penalty_mask = penalty_mask.to(sample_location_front_reshape.device).to(sample_location_front_reshape.dtype)
            penalty_mask = penalty_mask.reshape(-1)
            min_diff = min_diff_in_roi * penalty_mask
            cur_loss = torch.mean(min_diff)
        else:
            cur_loss = torch.mean(min_diff)
        loss +=  cur_loss
        layer_count += 1
    loss = loss / layer_count
    return loss


def roi_attention_loss_func(patch_mask,  sampling_locations_list, attention_weights_logit_list, volum_size_list,volume_mask_list, indexes_list, in_roi=False, is_enhance=True):
    if attention_weights_logit_list is None or len(attention_weights_logit_list) == 0:
        return 0
    loss = 0
    for i, attention_weight in enumerate(attention_weights_logit_list):
        sample_location = sampling_locations_list[i]
        attention_weight_front = attention_weights_logit_list[i][0]
        sample_location_front = sample_location[0]
        sample_location_front_reshape = sample_location_front.reshape(-1,1,2)
        is_inside_mask = create_point_mask_from_patch(patch_mask, sample_location_front_reshape)
        attention_weight_front_reshape = attention_weight_front.reshape(-1)
        valid_attention_weight = attention_weight_front_reshape * is_inside_mask
        if is_enhance:
            loss +=  -torch.mean(valid_attention_weight)
        else:
            loss +=  torch.mean(valid_attention_weight)

    loss = loss / float(len(attention_weights_logit_list))
    return loss


def loss_with_type(loss_type, pred_occ, voxel_softmax_max_masked, voxel_softmax, target_rank_mask,a1, a2):
    if loss_type == LossType.softmax_reduce_other_sum_log:
        expanded_mask = voxel_softmax_max_masked.unsqueeze(0).expand(17, -1, -1, -1)
        voxel_softmax = voxel_softmax[0] * expanded_mask
        c, h, w, z = voxel_softmax.shape
        car_scores = voxel_softmax[4]
        other_channels = torch.cat([voxel_softmax[:4], voxel_softmax[5:]], dim=0)
        if c == 1:
            other_sum = torch.zeros_like(torch.sum(other_channels, dim=0))
        else:
            other_sum = torch.sum(other_channels, dim=0) / (c - 1)
        car_scores = car_scores - other_sum
        valid_mask = car_scores >= 0
        valid_car_scores = car_scores[valid_mask]
        loss = torch.log(torch.sum(valid_car_scores) + 1e-8)
    elif loss_type == LossType.softmax_empty_road:
        eps = torch.tensor(1e-8, dtype=voxel_softmax_max_masked.dtype).to(voxel_softmax_max_masked.device)
        voxel_softmax = torch.clamp(voxel_softmax, 1e-8, 1 - 1e-8)
        expanded_mask = voxel_softmax_max_masked.unsqueeze(0).expand(17, -1, -1, -1)
        voxel_softmax_masked = voxel_softmax[0] * expanded_mask
        c, h, w, z = voxel_softmax_masked.shape
        car_scores = voxel_softmax_masked[4]
        empty_scores = voxel_softmax_masked[0]
        road_scores = voxel_softmax_masked[11]
        car_scores_prime = car_scores - a1 * empty_scores - a2 * road_scores
        loss = torch.sum(car_scores_prime) / (target_rank_mask.sum() + eps)
    elif loss_type == LossType.softmax_empty_road_max:
        expanded_mask = voxel_softmax_max_masked.unsqueeze(0).expand(17, -1, -1, -1)
        voxel_softmax_masked = voxel_softmax[0] * expanded_mask
        c, h, w, z = voxel_softmax_masked.shape
        car_scores = voxel_softmax_masked[4]
        empty_scores = voxel_softmax_masked[0]
        road_scores = voxel_softmax_masked[11]
        car_scores_prime = car_scores - torch.max(empty_scores, road_scores)
        car_scores_prime[car_scores_prime < 0] = 0
        loss = torch.sum(car_scores_prime) / (target_rank_mask.sum() + 1e-8)
    elif loss_type == LossType.logit_empty_road_max:
        expanded_mask = voxel_softmax_max_masked.unsqueeze(0).expand(17, -1, -1, -1)
        voxel_softmax_masked = pred_occ[0] * expanded_mask
        c, h, w, z = voxel_softmax_masked.shape
        car_scores = voxel_softmax_masked[4]
        empty_scores = voxel_softmax_masked[0]
        road_scores = voxel_softmax_masked[11]
        car_scores_prime = car_scores - torch.max(empty_scores, road_scores)
        car_scores_prime[car_scores_prime < 0] = 0
        loss = torch.sum(car_scores_prime) / (target_rank_mask.sum() + 1e-8)
    elif loss_type == LossType.logit_softplus_empty_road_max:
        expanded_mask = voxel_softmax_max_masked.unsqueeze(0).expand(17, -1, -1, -1)
        voxel_softmax_masked = pred_occ[0] * expanded_mask
        c, h, w, z = voxel_softmax_masked.shape
        car_scores = voxel_softmax_masked[4]
        empty_scores = voxel_softmax_masked[0]
        road_scores = voxel_softmax_masked[11]
        car_scores_prime = car_scores - torch.max(empty_scores, road_scores)
        car_scores_prime[car_scores_prime < 0] = 0
        softplus_ret = torch.nn.functional.softplus(car_scores_prime) - torch.log(torch.tensor(2.0))
        loss = torch.sum(softplus_ret) / (target_rank_mask.sum() + 1e-8)
    elif loss_type == LossType.logit_softplus_empty_road:
        expanded_mask = voxel_softmax_max_masked.unsqueeze(0).expand(17, -1, -1, -1)
        voxel_softmax_masked = pred_occ[0] * expanded_mask
        c, h, w, z = voxel_softmax_masked.shape
        car_scores = voxel_softmax_masked[4]
        empty_scores = voxel_softmax_masked[0]
        road_scores = voxel_softmax_masked[11]
        car_scores_prime = car_scores - a1 * empty_scores - a2 * road_scores
        ret_soft_plus = torch.nn.functional.softplus(car_scores_prime) - torch.log(torch.tensor(2.0))
        loss = torch.sum(ret_soft_plus) / (target_rank_mask.sum() + 1e-8)
    elif loss_type == LossType.softmax_softplus_empty_road:
        expanded_mask = voxel_softmax_max_masked.unsqueeze(0).expand(17, -1, -1, -1)
        voxel_softmax_masked = voxel_softmax[0] * expanded_mask
        c, h, w, z = voxel_softmax_masked.shape
        car_scores = voxel_softmax_masked[4]
        empty_scores = voxel_softmax_masked[0]
        road_scores = voxel_softmax_masked[11]
        car_scores_prime = car_scores - a1 * empty_scores - a2 * road_scores
        ret_soft_plus = torch.nn.functional.softplus(car_scores_prime) - torch.log(torch.tensor(2.0))
        loss = torch.sum(ret_soft_plus) / (target_rank_mask.sum() + 1e-8)
    elif loss_type == LossType.softmax_softplus_empty_road_max:
        expanded_mask = voxel_softmax_max_masked.unsqueeze(0).expand(17, -1, -1, -1)
        voxel_softmax_masked = voxel_softmax[0] * expanded_mask
        c, h, w, z = voxel_softmax_masked.shape
        car_scores = voxel_softmax_masked[4]
        empty_scores = voxel_softmax_masked[0]
        road_scores = voxel_softmax_masked[11]
        car_scores_prime = car_scores - torch.max(empty_scores, road_scores)
        car_scores_prime[car_scores_prime < 0] = 0
        softplus_ret = torch.nn.functional.softplus(car_scores_prime) - torch.log(torch.tensor(2.0))
        loss = torch.sum(softplus_ret) / (target_rank_mask.sum() + 1e-8)
    elif loss_type == LossType.logit_softplus_paper:

        pred = pred_occ[0]
        z_non = pred[0]
        z_drv = pred[11]
        w_sum = a1 + a2
        w1 = a1 / w_sum
        w2 = a2 / w_sum
        non_target = torch.cat([pred[1:11], pred[12:]], dim=0)
        delta = torch.logsumexp(non_target, dim=0) - w1 * z_non - w2 * z_drv
        ell = torch.nn.functional.softplus(delta)
        loss = torch.sum(ell * voxel_softmax_max_masked) / (target_rank_mask.sum() + 1e-8)
    else:
        loss = 0


    return loss

def loss_with_roi(loss_type, pred_occ, width=2.0, a1=1.0, a2=0.8):
    roi_mask = get_roi_mask(width)
    roi_mask = roi_mask.to(pred_occ.device)

    voxel_softmax = torch.softmax(pred_occ, dim=1)
    voxel_softmax_max_score, voxel_softmax_max = torch.max(voxel_softmax, dim=1)
    voxel_softmax_max = voxel_softmax_max[0]


    unsuccessful = (voxel_softmax_max != 0) & (voxel_softmax_max != 11)
    voxel_softmax_max_masked = (roi_mask * unsuccessful.to(roi_mask.device)).to(voxel_softmax.dtype)

    voxel_count = torch.count_nonzero(voxel_softmax_max_masked).item()

    loss = loss_with_type(loss_type, pred_occ, voxel_softmax_max_masked, voxel_softmax, roi_mask, a1=a1, a2=a2)

    return loss, voxel_count, voxel_softmax_max


def loss_with_detect(loss_type, pred_occ, a1=1.0, a2=0.8):
    voxel_softmax = torch.softmax(pred_occ, dim=1)
    voxel_softmax_max_score, voxel_softmax_max = torch.max(voxel_softmax, dim=1)
    voxel_softmax_max = voxel_softmax_max[0]
    cars_info_list = detect_all_cars_with_ret(voxel_softmax_max)
    selected_car = detect_valid_car_in_range(cars_info_list)
    if selected_car is not None:
        range_mask = selected_car.get_3dmask().to(pred_occ.device)


        unsuccessful = (voxel_softmax_max != 0) & (voxel_softmax_max != 11)
        voxel_softmax_max_masked = (range_mask * unsuccessful.to(range_mask.device)).to(voxel_softmax.dtype)

        vexel_count = torch.count_nonzero(voxel_softmax_max_masked).item()
        loss = loss_with_type(loss_type, pred_occ, voxel_softmax_max_masked, voxel_softmax, range_mask, a1=a1, a2=a2)

        return loss, vexel_count, voxel_softmax_max
    else:
        return 0, 0, voxel_softmax_max
