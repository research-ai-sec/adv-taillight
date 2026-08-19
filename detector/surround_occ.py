from mmcv import Config
import os
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import torch
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.cnn import fuse_conv_bn
import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor

import torch.nn.functional as F
import random
import torchvision
from utils.image_tool import generate_positions, get_mask_bounds, calulate_scale_factor, load_all_from_vir_car_dir, max_brightness_preserve_contrast
import math
from collections import deque

class UniformSampler:
    def __init__(self, items):
        self.items = deque(items)
        self.reset()

    def reset(self):

        random.shuffle(self.items)
        self.items.rotate(0)

    def sample(self, n):

        sampled = []
        for _ in range(n):
            if not self.items:
                self.reset()
            sampled.append(self.items[0])
            self.items.rotate(-1)
        return sampled


def get_filename_vir_car(h, d, r):
    file_name = f"h_{h:.2f}_d_{d:.2f}_r_{r:.2f}"
    return file_name


def perspective_transform_pytorch(ref_points, points1, c_ret_img):


    if c_ret_img.dim() == 3:
        c_ret_img = c_ret_img.unsqueeze(0)


    ref_points = torch.tensor(ref_points, dtype=torch.float32).view(-1, 2)
    points1 = torch.tensor(points1, dtype=torch.float32).view(-1, 2)

    warped_img = torchvision.transforms.functional.perspective(c_ret_img, ref_points, points1, fill=0)

    return warped_img


def composite_images_new(mask, imgs_tensor, is_bgr=False, scale=1.0, n_gaps=0, max_offset=0.5, points=None):

    def tensor2int(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.item()

        return int(round(tensor))


    x_min, x_max, y_min, y_max = get_mask_bounds(mask)
    x_min = tensor2int(x_min)
    x_max = tensor2int(x_max)
    y_min = tensor2int(y_min)
    y_max = tensor2int(y_max)


    img_target_size = y_max - y_min + 1
    canvas_width  = x_max - x_min + 1
    canvas_height = img_target_size
    canvas = torch.zeros((3, canvas_height, canvas_width ),
                        dtype=torch.float32, device=mask.device)

    if points is not None:
        x_points = points
    else:
        if n_gaps == 0:
            num_imgs = len(imgs_tensor)
            total_img_width = img_target_size * num_imgs
            start_offset = (canvas_width - total_img_width) // 2
            x_points = [start_offset + i * img_target_size for i in range(num_imgs)]
        else:
            x_points = generate_positions(num_images=len(imgs_tensor),image_width=img_target_size,total_width=canvas_width,max_offset_ratio=max_offset, max_attempts=100)


    for idx, simg in enumerate(imgs_tensor):
        if is_bgr:
            b = simg[0,:,:]
            g = simg[1,:,:]
            r = simg[2,:,:]
            b = torch.zeros_like(r)
            bgr = torch.stack([b, g, r], dim=0)
        else:
            r = simg[0,:,:]
            g = simg[1,:,:]
            b = torch.zeros_like(r)
            bgr = torch.stack([b, g, r], dim=0)
        scale_img = bgr.unsqueeze(0)
        if scale != 1.0:
            mid_height = math.ceil(img_target_size * scale)
            mid_width  = math.ceil(img_target_size * scale)
            scale_img = torch.nn.functional.interpolate(scale_img, size=(mid_height, mid_width), mode='area')
            scale_img = torch.nn.functional.interpolate(scale_img, size=(img_target_size, img_target_size), mode='bilinear')
        else:
            scale_img = torch.nn.functional.interpolate(scale_img, size=(img_target_size, img_target_size), mode='bilinear')
        bgr = scale_img[0]

        pos_x = int(x_points[idx])
        end_x = pos_x + img_target_size

        if pos_x < 0 and end_x < 0:
            continue

        if pos_x > x_max and end_x > x_max:
            continue

        if pos_x < 0:
            pos_x = 0
        if end_x > x_max:
            end_x = x_max

        if end_x > canvas_width:
            bgr = bgr[:, :, :(canvas_width - pos_x)]
            end_x = canvas_width

        if pos_x < canvas_width and end_x > pos_x:
            canvas[:, :, pos_x:end_x] = bgr[:, :, :(end_x - pos_x)]


    padded_img = torch.zeros_like(mask, dtype=torch.float32)
    if len(padded_img.shape) == 5:
        padded_img[0, :, :, y_min:y_min+canvas_height, x_min:x_min+canvas_width] = canvas
    elif len(padded_img.shape) == 4:
        padded_img[:, :, y_min:y_min+canvas_height, x_min:x_min+canvas_width] = canvas
    elif len(padded_img.shape) == 3:
        padded_img[:, y_min:y_min+canvas_height, x_min:x_min+canvas_width] = canvas
    return padded_img, x_points, canvas


def get_corner_points(mask):
    x_min, x_max, y_min, y_max = get_mask_bounds(mask)

    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def generate_light_aug_params():

    alpha = random.uniform(0.7, 1.3)
    beta = random.uniform(-30, 30)

    return alpha, beta


def prepare_with_fake_img(use_light_aug, car_info, ref_mask_tensor,  fimg_list, bg_img, with_scale=False, is_bgr=False, n_gaps=0, max_offset=0.5, alpha=None, beta=None, points=None):


    car_img, car_mask, screen_mask, rect_mask, ring_img, ring_mask = car_info
    if with_scale:
        scale_factor = calulate_scale_factor(ref_mask_tensor)
    else:
        scale_factor = 1.0

    if fimg_list is not None:
        padded_img, x_points,  canvas = composite_images_new(ref_mask_tensor, fimg_list, is_bgr, scale_factor, n_gaps, max_offset, points)
        if len(padded_img.shape) == 5:
            padded_img = padded_img[0][0]


        ref_points = get_corner_points(ref_mask_tensor)
        tar_points = get_corner_points(rect_mask)
        warped_img = perspective_transform_pytorch(ref_points, tar_points, padded_img)[0]
        warped_img = warped_img.to(bg_img.device)

    result_car_bg = car_img * car_mask + bg_img[0][0] * (1-car_mask)
    result_car_bg = result_car_bg * (1-ring_mask) + max_brightness_preserve_contrast(ring_img)

    if fimg_list is not None:
        result = warped_img * screen_mask + result_car_bg * (1 - screen_mask)
        bg_img[0][0] = result
    else:
        bg_img[0][0] = result_car_bg


    if use_light_aug:
        if alpha is None:
            alpha, beta = generate_light_aug_params()
        bg_img = torch.clamp(alpha * bg_img + beta, 0, 255)

    return bg_img, screen_mask, car_mask

def pad_to_multiple_tensor(img_tensor, size_divisor=32, pad_val=0):

    if img_tensor.dim() == 3:
        h, w = img_tensor.shape[-2], img_tensor.shape[-1]
    elif img_tensor.dim() == 4:
        h, w = img_tensor.shape[-2], img_tensor.shape[-1]
    elif img_tensor.dim() == 5:
        h, w = img_tensor.shape[-2], img_tensor.shape[-1]
    else:
        raise ValueError(f"Unsupported tensor dimension: {img_tensor.dim()}. Expected 3, 4, or 5 dimensions.")
    pad_h = (size_divisor - h % size_divisor) % size_divisor
    pad_w = (size_divisor - w % size_divisor) % size_divisor

    if pad_h > 0 or pad_w > 0:

        padding = (0, pad_w, 0, pad_h)
        img_tensor = F.pad(img_tensor, padding, value=pad_val)

    return img_tensor


@PIPELINES.register_module()
class CarImagePipeline(object):
    def __init__(self, h_list, d_list, r_list, size_divisor, mean, std, device, to_rgb=True, strict_uniform=False):
        self.h_list = h_list
        self.d_list = d_list
        self.r_list = r_list
        self.size_divisor =  size_divisor if size_divisor else 32
        self.mean_list = mean
        self.std_list = std
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.strict_uniform = strict_uniform

        current_file_path = Path(__file__).resolve()
        root_path = current_file_path.parent.parent
        src_dir = os.path.join(root_path, 'data', 'images', 'vir-car')
        ref_path = os.path.join(root_path, 'data', 'images', 'ref.png')
        print(f'ref_path:{ref_path}')
        ref_mask = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        _, ref_mask = cv2.threshold(ref_mask, 127, 255, cv2.THRESH_BINARY)
        ref_mask_tensor_org = torch.from_numpy(ref_mask).float() / 255.0
        ref_mask_tensor = torch.stack([ref_mask_tensor_org] * 3, dim=0).unsqueeze(0).unsqueeze(0).to(device)
        self.ref_mask_tensor = ref_mask_tensor
        if (h_list is not None) and (d_list is not None) and (r_list is not None):
            pos_key_list = []
            pos_map = {}
            for h in h_list:
                for d in d_list:
                    for r in r_list:


                        file_name = get_filename_vir_car(h,d,r)
                        pos_key_list.append(file_name)
                        car_info = load_all_from_vir_car_dir(h, d, r, src_dir, file_name, device=device, to_tensor=True)
                        pos_map[file_name] = car_info
            print(f'pos_key_list={pos_key_list}')
            self.pos_key_list = pos_key_list
            self.pos_map = pos_map
            self.unifor_sample = UniformSampler(pos_key_list)
        else:
            self.pos_key_list = None
            self.pos_map = None

    def __call__(self, results):
        img = results['img']
        for i in range(len(img)):
            img[i] = img[i].astype(np.float32)
        results['img'] = img
        results = self._load_occupancy(results)
        results = self._normalize_img(results)
        results = self._pad_img(results)
        results = self._masked_car_with_imgs(results)
        results = self.format_bundle(results)
        if self.pos_key_list is not None:
            if self.strict_uniform == True:
                selected_key = self.unifor_sample.sample(1)[0]
            else:
                selected_key = random.choice(self.pos_key_list)
            car_info = self.pos_map[selected_key]
            results['car_info'] = car_info
            results['ref_mask'] = self.ref_mask_tensor
            results['offset'] = selected_key
        for key in results.keys():
            value = results[key]
            if value is None:
                print(f'{key} is None')
        return results


    def get_lidar_points_with_gt_bbox(self, gt_names, gt_boxes):
        car_list_3d_points_list = []
        for i in range(len(gt_names)):
            gname = gt_names[i]
            if gname == 'car':
                gbbox = gt_boxes[i]
                x,y,z, w, l, h, theta = gbbox
                center_point = np.array([x, y, z])
                corners_local = np.array([
                    [ -w/2, -l/2,   -h/2],
                    [ w/2, -l/2,  -h/2],
                    [ w/2, -l/2,    h/2],
                    [ -w/2, -l/2,   h/2],
                    [ -w/2, l/2,   -h/2],
                    [ w/2, l/2,  -h/2],
                    [ w/2, l/2,    h/2],
                    [ -w/2, l/2,   h/2],
                ])
                corner_points = corners_local + center_point
                car_list_3d_points_list.append(corner_points)

        return car_list_3d_points_list

    def trans_lidar_to_img(self, lidar2img_rt, points):

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

    def _masked_car_with_imgs(self, results):
        imgs = results['img']
        lidar2img_list = results['lidar2img']
        gt_names= results['gt_names']
        gt_boxes= results['gt_boxes']
        gt_point_list = self.get_lidar_points_with_gt_bbox(gt_names, gt_boxes)
        masked_img_list = []
        p_img_total = []
        camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
        for i in range(len(imgs)):
            cam_type = camera_types[i]
            lidar2img_rt = lidar2img_list[i]
            img = imgs[i]
            mask_img = np.zeros_like(img)
            p_img_list = []
            for gt_points in gt_point_list:
                p_img = self.trans_lidar_to_img(lidar2img_rt, gt_points)
                if p_img is not None:
                    p_img_list.append(p_img)
                    corners_2d = p_img.reshape(-1, 2).astype(int)
                    x_min, x_max = np.min(corners_2d[:, 1]), np.max(corners_2d[:, 1])
                    y_min, y_max = np.min(corners_2d[:, 0]), np.max(corners_2d[:, 0])
                    mask_img[x_min:x_max+1, y_min:y_max+1] = 1
            if np.count_nonzero(mask_img) > 0:
                masked_img_list.append(mask_img)
            else:
                masked_img_list.append(np.zeros_like(mask_img))
            p_img_total.append(p_img_list)
        results['gt_b_points'] = p_img_total
        results['mask'] = masked_img_list
        return results


    def _load_occupancy(self, results):
        results['gt_occ'] = ''
        return results


        occ_path = results['occ_path']
        occ_path = occ_path.replace("nuScenes/v1.0-trainval/", "nuscenes_occ/")
        occ_path = occ_path.replace('LIDAR_TOP/', '')
        occ_path = occ_path + ".npy"        
        if os.path.exists(occ_path):
            occ = np.load(occ_path)
            occ = occ.astype(np.float32)


            if self.use_semantic:
                occ[..., 3][occ[..., 3] == 0] = 255
            else:
                occ = occ[occ[..., 3] > 0]
                occ[..., 3] = 1

            results['gt_occ'] = occ
        else:
            results['gt_occ'] = np.array(0)
        return results

    def _normalize_img(self, results):
        if self.to_rgb:
            results['img'] = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def _pad_img(self, results):
        h,w,c = results['img'][0].shape
        if self.size_divisor is not None:
            h_pad = math.ceil(h / self.size_divisor) * self.size_divisor
            w_pad = math.ceil(w / self.size_divisor) * self.size_divisor
        else:
            h_pad = h
            w_pad = w
        pad_shape = (h_pad,w_pad, c)
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = results['img']
        results['img_shape'] = [pad_shape for _ in results['img']]
        results['pad_shape'] = [pad_shape for _ in results['img']]
        results['pad_fixed_size'] = self.size_divisor
        results['pad_size_divisor'] = self.size_divisor
        return results

    def format_bundle(self, results):
        imgs = [img.transpose(2, 0, 1) for img in results['img']]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        results['img'] = DC(to_tensor(imgs), stack=True)


        masks = [img.transpose(2, 0, 1) for img in results['mask']]
        masks = np.ascontiguousarray(np.stack(masks, axis=0))
        results['mask'] = DC(to_tensor(masks), stack=True)

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))

        return results


color_map = np.array(
        [
            [0, 0, 0, 255],
            [255, 120, 50, 255],
            [255, 192, 203, 255],
            [255, 255, 0, 255],
            [0, 150, 245, 255],
            [0, 255, 255, 255],
            [200, 180, 0, 255],
            [255, 0, 0, 255],
            [255, 240, 150, 255],
            [135, 60, 0, 255],
            [160, 32, 240, 255],
            [255, 0, 255, 255],

            [139, 137, 137, 255],
            [75, 0, 75, 255],
            [150, 240, 80, 255],
            [230, 230, 250, 255],
            [0, 175, 0, 255],
        ]
    )


def prepare_with_cfg(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.get('cudnn_benchmark', False) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None


    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = fuse_conv_bn(model)
    if torch.cuda.is_available():
        model = model.cuda(device)

    return model, cfg