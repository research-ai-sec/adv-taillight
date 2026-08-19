import torch


def pc_grad_project_(grads):
    proj = [g.clone() for g in grads]
    num_tasks = len(grads)


    dot = torch.sum(proj[0] * proj[1])
    if dot < 0:
        norm_sq = torch.sum(proj[0] * proj[0])
        if norm_sq > 1e-8:
            proj[1] = proj[1] - (dot / norm_sq) * proj[0]

    if num_tasks == 3:

        dot = torch.sum(proj[0] * proj[2])
        if dot < 0:
            norm_sq = torch.sum(proj[0] * proj[0])
            if norm_sq > 1e-8:
                proj[2] = proj[2] - (dot / norm_sq) * proj[0]

    return proj


def compute_statistic(name, tensor_list):
    if tensor_list is not None:
        total_norm = 0.0
        total_val = 0.0
        total_mean = 0.0
        maxv = None
        minv = None
        has_nan = False
        has_inf = False
        num_layers = len(tensor_list)
        for i in range(num_layers):
            if tensor_list[i] is None:
                continue
            if torch.isnan(tensor_list[i]).any():
                has_nan = True
            if torch.isinf(tensor_list[i]).any():
                has_inf = True
            total_norm += torch.norm(tensor_list[i]).item()
            total_val += torch.var(tensor_list[i]).item()
            total_mean += torch.mean(tensor_list[i]).item()

            cur_max = torch.max(torch.abs(tensor_list[i])).item()
            cur_min = torch.min(torch.abs(tensor_list[i])).item()
            if maxv is None or cur_max > maxv:
                maxv = cur_max
            if minv is None or cur_min < minv:
                minv = cur_min
        norm_point = total_norm / num_layers
        grad_variance_point =  total_val / num_layers
        mean_point = total_mean  / num_layers
    else:
        norm_point = 0
        grad_variance_point = 0
        mean_point = 0
        maxv = 0
        minv = 0
        has_nan = False
        has_inf = False

    info = {f'{name}-norm': norm_point, f'{name}-min': minv, f'{name}-max': maxv, f'{name}-var': grad_variance_point, f'{name}-mean': mean_point,
        f'{name}-has_nan': has_nan,
        f'{name}-has_inf': has_inf}
    return info


def handle_project_grad(name, grad_to_project, grad_adv_temp, final_grad_list, att_weight):
    num_layers = len(grad_adv_temp)
    if grad_to_project is not None:
        total_cos_sim = 0.0
        for i in range(num_layers):


            grad_adv_flat = grad_adv_temp[i].flatten()
            grad_point_flat = grad_to_project[i].flatten()


            cos_sim = torch.nn.functional.cosine_similarity(
                grad_adv_flat.unsqueeze(0),
                grad_point_flat.unsqueeze(0),
                dim=1
            ).item()

            total_cos_sim += cos_sim


            grad_point_cur = grad_to_project[i]

            if cos_sim < 0:
                dot = torch.dot(grad_adv_flat, grad_point_flat)
                norm_sq = torch.norm(grad_adv_flat) ** 2
                if norm_sq > 1e-6:


                    projection = (dot / norm_sq) * grad_adv_temp[i]
                    grad_point_cur = grad_to_project[i] - projection


            final_grad_list[i] += att_weight * grad_point_cur
        cos_sim_adv_point = total_cos_sim / num_layers
    else:
        cos_sim_adv_point = 0

    info = { f'{name}-cos_sim': cos_sim_adv_point}
    return info


def compute_pc_grad(grad_adv_temp, grad_point_temp, grad_att_temp, other_weight=1.0):

    if grad_adv_temp is None or len(grad_adv_temp) == 0:
        print(f'adv grad is None')
        return None, None

    info1 = compute_statistic('adv', grad_adv_temp)
    if grad_point_temp is None and grad_att_temp is None:
        return grad_adv_temp, info1


    if grad_att_temp is not None:
        weights_list = [1.0, other_weight, other_weight]
        total = sum(weights_list)
        normalized_list = [x / total for x in weights_list]
        adv_weight, points_weight, att_weight = normalized_list
    else:
        weights_list = [1.0, other_weight]
        total = sum(weights_list)
        normalized_list = [x / total for x in weights_list]
        adv_weight, points_weight = normalized_list
        att_weight = 0
    final_grad_list = [grad.clone() * adv_weight  for grad in grad_adv_temp]
    info12 = compute_statistic('poinnt', grad_point_temp)
    info2 = handle_project_grad("point", grad_point_temp, grad_adv_temp, final_grad_list, points_weight)
    for key,value in info2.items():
        info1[key] = value
    info13 = compute_statistic('att', grad_att_temp)
    info3 = handle_project_grad("att", grad_att_temp, grad_adv_temp, final_grad_list, att_weight)


    merge = info1 | info12 | info2 | info13 | info3

    merge_stats = info1.copy()
    merge_stats.update(info12)
    merge_stats.update(info2)
    merge_stats.update(info13)
    merge_stats.update(info3)

    return final_grad_list, merge