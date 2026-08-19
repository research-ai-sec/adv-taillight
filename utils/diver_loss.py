import torch
from detector.voxel_car_detector import get_roi_mask, downsample_3d_any


def compute_feature_diversity(feature_map: torch.Tensor, dim: int = -2) -> torch.Tensor:


    z0 = feature_map.mean(dim=dim, keepdim=True)  


    residual = feature_map - z0


    if feature_map.dim() == 3:
        r = torch.norm(residual, p='fro', dim=(1, 2))
    elif feature_map.dim() == 2:
        r = torch.norm(residual, p='fro')
    else:
        raise ValueError(f"Unsupported feature_map shape. Expected 2D or 3D tensor. {feature_map.shape}")

    return r


def diver_loss_with_tensor(tensor, beta, dim=0):
    eps = 1e-10

    r = compute_feature_diversity(tensor, dim=dim)          
    c = tensor.shape[-1]
    n_r = tensor.shape[dim]
    sigma = r / torch.sqrt(torch.tensor(float(c) * float(n_r), device=tensor.device, dtype=tensor.dtype))

    cur_loss = (beta * torch.log(1.0 + sigma)).mean()
    return cur_loss

def tensor_with_roi(tensor, roi_mask, w,h,z, c, bs=1):
    roi_mask_resize = downsample_3d_any(roi_mask, target_size=(w, h,  z)).to(tensor.device).to(tensor.dtype).unsqueeze(0).unsqueeze(-1).repeat(bs,1,1,1,1)
    feat_flat = tensor.reshape(-1, c)
    roi_mask_flat = roi_mask_resize.reshape(-1).bool()
    selected_features_roi = feat_flat[roi_mask_flat]
    return selected_features_roi


def downsample_3d_any_fast(tensor, target_size=(100, 100, 8)):

    w,h, z = tensor.shape
    target_w, target_h, target_z = target_size

    if h % target_h != 0 or w % target_w != 0 or z % target_z != 0:
         raise ValueError("Dimensions must be divisible by target sizes")

    factor_h = h // target_h
    factor_w = w // target_w
    factor_z = z // target_z


    x = tensor.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3)


    kernel_size = (factor_z, factor_h, factor_w)
    stride = (factor_z, factor_h, factor_w)


    pooled = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=stride)


    result = pooled.squeeze(0).squeeze(0).permute(2, 1, 0)

    return result.float()

def tensor_with_roi_optimized(tensor, roi_mask, w, h, z, c, bs=1):


    if not isinstance(roi_mask, torch.Tensor):
         roi_mask = torch.tensor(roi_mask)
    assert tensor.shape[0] == 1, "this simplified version only supports bs=1"
    tensor_squeezed = tensor[0]

    roi_resized = downsample_3d_any_fast(roi_mask, target_size=(w, h, z))
    roi_bool = roi_resized.to(tensor.device).bool()


    w_idx, h_idx, z_idx = roi_bool.nonzero(as_tuple=True)

    result = tensor_squeezed[w_idx, h_idx, z_idx, :]

    return result


def diver_loss_with_model(outputs, beta=1.0):
    pred_occ = outputs['pred_occ']
    roi_mask = get_roi_mask().to(pred_occ.device).to(pred_occ.dtype)


    att_block_value = outputs['att_block_value']
    att_layer_value = outputs['att_layer_value']
    conv_loss = 0
    norm1_loss = 0
    norm2_loss = 0
    ffn_loss = 0
    layer_out_loss = 0

    for layer_idx in range(10):
        conv = att_block_value['conv'][layer_idx]
        bs, c, w,h, z = conv.shape


        if layer_idx in [0]:
            norm1 = att_block_value['norm1'][layer_idx]
            norm1_att_resize = norm1.reshape(bs, z, h, w, -1).permute(0,3,2,1,4)
            norm_roi = tensor_with_roi_optimized(norm1_att_resize, roi_mask, w,h,z, c, bs)
            norm1_loss += diver_loss_with_tensor(norm_roi,  beta=beta, dim=0)


        if layer_idx in [0]:
            ffn = att_block_value['ffn'][layer_idx]
            ffn_att_resize = ffn.reshape(bs, z, h, w, -1).permute(0,3,2,1,4)
            ffn_roi = tensor_with_roi_optimized(ffn_att_resize, roi_mask, w,h,z, c, bs)
            ffn_loss += diver_loss_with_tensor(ffn_roi,  beta=beta, dim=0)

        if layer_idx in [0,1,2,3]:
            norm2 = att_block_value['norm2'][layer_idx]
            norm2_att_resize = norm2.reshape(bs, z, h, w, -1).permute(0,3,2,1,4)
            norm2_roi = tensor_with_roi_optimized(norm2_att_resize, roi_mask, w,h,z, c, bs)
            norm2_loss += diver_loss_with_tensor(norm2_roi,  beta=beta, dim=0)


        if layer_idx in [0,1,2,3]:
            conv_reshape = conv.permute(0,2,3,4,1)
            conv_roi = tensor_with_roi_optimized(conv_reshape, roi_mask, w,h,z, c, bs)
            conv_loss += diver_loss_with_tensor(conv_roi,  beta=beta, dim=0)

        if layer_idx in [0,1,2,3]:
            layer_out = att_layer_value[layer_idx]
            layer_out_reshape = layer_out.reshape(bs, z,h,w, c).permute(0,3,2,1,4)
            layer_out_roi = tensor_with_roi_optimized(layer_out_reshape, roi_mask, w,h,z, c, bs)
            layer_out_loss += diver_loss_with_tensor(layer_out_roi,  beta=beta, dim=0)


    num_layers = 1 + 1 + 4 + 4 + 4
    total_loss = (norm1_loss + ffn_loss + norm2_loss + conv_loss + layer_out_loss) / num_layers

    return total_loss
