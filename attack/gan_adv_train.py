import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from enum import Enum
from pathlib import Path
from collections import deque
import time
import json
import csv
import gc
import copy
import random

import numpy as np
import torch
import cv2
import dill
import click
import dnnlib
from dnnlib import EasyDict
from loguru import logger
from mmcv import Config
from mmdet3d.datasets import build_dataset
from SurroundOcc.projects.mmdet3d_plugin.datasets.builder import build_dataloader

from attack.styleganxl import load_generator_styleganxl, get_w_from_seed
from detector.surround_occ import (
    prepare_with_cfg,
    prepare_with_fake_img,
    pad_to_multiple_tensor,
    generate_light_aug_params,
)
from detector.voxel_car_detector import get_roi_mask
from spocc.surround_occ_model import SurroundOccModel
from utils.scene_utils import get_scene_list
from utils.loss_func import LossType, AttentionLossType, loss_with_roi, roi_point_in_loss_func, roi_attention_loss_func
from utils.diver_loss import diver_loss_with_model
from utils.pc_grad import compute_pc_grad

out_dir = os.path.join(_ROOT, 'output_new', 'adv_train_amp')
os.makedirs(out_dir, exist_ok=True)
logger.add(os.path.join(out_dir, "error.log"), level="ERROR", backtrace=True, diagnose=True)


def exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Uncaught exception occurred")
sys.excepthook = exception_handler


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


device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainParam():

    def __init__(self, args):
        self.args = args

    def args(self):
        return self.args


class PerLossType(Enum):
    img_l2 = 'img_l2'
    d0_l1 = "d0_l1"
    vgg = "vgg"
    d = "d"
    none = "none"

class WGenMode(Enum):
    once = "once"
    epoch = "epoch"
    batch = "batch"
    img  = "img"
    scene_start = "scene_start"

class InitType(Enum):
    zero = 'zero'
    random = 'random'


class GanAdvTrainer:
    def __init__(self,model_occ, device, c_dim, out_dir, loss_type, loss_att, cfg=None, D=None, d_feature=None, bb_name=None):
        self.model_occ = model_occ
        self.org_generator = None
        self.generator = None
        self.model = None
        self.device = device
        self.d_feature = d_feature
        self.bb_name = bb_name
        self.loss_type = loss_type
        self.att_loss_type = loss_att
        self.D = D.to(device) if D is not None else None
        self.c_dim = c_dim
        self.mean = [103.530, 116.280, 123.675]
        self.std = [1.0, 1.0, 1.0]
        self.size_divisor = 32
        self.pad_val = 0
        self.class_sample = UniformSampler([i for i in range(c_dim)])


        self.train_params = EasyDict({
            'loss_type': LossType.logit_softplus_paper,
            'init_type': InitType.zero,
            'resume_checkpoint': None
        })
        if cfg is not None:
            self._update_params_from_cfg(cfg)
        self.use_light_aug = self.train_params.use_light_aug

        log_path = os.path.join(out_dir, "training.log")

        logger.add(log_path, format="{message}", level="INFO", enqueue=True)


    def _update_params_from_cfg(self, cfg):

        for key, value in cfg.items():

            if key == 'betas' and isinstance(value, (list, tuple)):
                self.train_params[key] = tuple(value)
            else:
                self.train_params[key] = value

    def sample_data(self, loader):
        while True:
            for batch in loader:
                yield batch


    def h_b_clip_func(self, rgb, max_h=30):
        rgb = rgb / 255.0
        r, g, b = rgb.unbind(dim=1)

        b = torch.zeros_like(b)
        new_rgb = torch.stack([r, g, b], dim=1)

        max_val, _ = torch.max(new_rgb, dim=1)
        min_val, _ = torch.min(new_rgb, dim=1)
        delta = max_val - min_val
        v = max_val
        s = torch.zeros_like(v)
        mask = (v != 0)
        s[mask] = delta[mask] / v[mask]
        h = torch.zeros_like(v)
        mask_r = (delta != 0) 
        h[mask_r] = 60 * ((g / delta)[mask_r] % 6) 
        h = h % 360

        h_clip = torch.clamp(h, 0, max_h)

        c = v * s
        h_ratio = h_clip / 60.0
        x = c * (1 - torch.abs(h_ratio % 2 - 1))
        m = v - c
        r = c + m
        g = x + m
        b = torch.zeros_like(r)
        rgb_ret = torch.stack([r, g, b], dim=1)

        rgb_ret = torch.clamp(rgb_ret, 0, 1)
        rgb_ret = rgb_ret * 255.0

        return rgb_ret


    def gaussian_distance_weights(self, region_width, sigma=1.0, device='cuda'):
        center = (region_width - 1) / 2
        x = torch.linspace(0, region_width - 1, region_width).to(device)
        weights = torch.exp(-(x - center)**2 / (2 * sigma**2))
        return weights


    def tensor_to_image(self, tensor, norm=True):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        if len(tensor.shape) == 5:
            tensor = tensor[0][0]
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        if tensor.shape[0] == 3:
            tensor = tensor.transpose(1, 2, 0)
        if norm:
            mean = np.array([103.530, 116.280, 123.675]).reshape(1, 1, 3)
            std = np.array([1.0, 1.0, 1.0]).reshape(1, 1, 3)
            tensor = (tensor * std + mean)
        img = tensor.astype(np.uint8)
        return img

    def get_w(self, generator):
        num_fake_img = self.train_params.num_fake_img
        z = torch.randn(size=(num_fake_img, generator.z_dim), device=device0)
        if generator.c_dim != 0:
            if self.train_params.eq_class:
                cid = self.class_sample.sample(1)[0]
                class_idx = torch.full((num_fake_img,), cid).to(device0).to(torch.long)
            else:
                class_idx = torch.randint(0, generator.c_dim, size=(num_fake_img,)).to(device0).to(torch.long)
        else:
            class_idx = None
        w = get_w_from_seed(generator, z, num_fake_img, device=device0, class_idx=class_idx)
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                cur_fake = generator.synthesis(w)
        return w, cur_fake

    def fake_to_model_input(self, cur_fake, car_info, ref_mask, bg_img, use_light_aug=False, alpha=None, beta=None):
        imgs = (cur_fake * 127.5 + 128).clamp(0, 255).to(torch.float32)
        imgs = self.h_b_clip_func(imgs, max_h=self.train_params.max_h)
        result_img_org, screen_mask, car_mask = prepare_with_fake_img(use_light_aug=use_light_aug,car_info=car_info, ref_mask_tensor=ref_mask, fimg_list=imgs, bg_img=bg_img, is_bgr=False, n_gaps=self.train_params.n_gaps, max_offset=self.train_params.max_offset, alpha=alpha, beta=beta)
        mean = torch.tensor(self.mean).reshape(1, 1, 3, 1, 1).to(bg_img.device)
        std = torch.tensor(self.std).reshape(1, 1, 3, 1, 1).to(bg_img.device)
        result_nom = (result_img_org - mean) / std
        combine_img = pad_to_multiple_tensor(result_nom, self.size_divisor, self.pad_val)
        ret_combine_img_org = self.tensor_to_image(result_img_org.detach().cpu().numpy()[0][0],norm=False)
        return combine_img, ret_combine_img_org, imgs, car_mask, screen_mask


    def save_image(self, out_idr, image_id, imgs, combine_image, prediction):


        if isinstance(combine_image, torch.Tensor):
            combine_image = combine_image.detach().cpu().numpy()
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.detach().cpu().numpy()
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        save_dir = os.path.join(out_idr, 'results')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{image_id:04d}-pred.npy"), prediction)
        cv2.imwrite(os.path.join(save_dir, f"{image_id:04d}.png"), combine_image)


    def save_checkpoint(self,total_fool, start_epoch, start_imgs, batch, generator, opts, resume_checkpoint):
        torch.save({
            'total_fool': total_fool,
            'epoch': start_epoch,
            'total_images': start_imgs+1,
            'batch': batch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': opts.state_dict()
        }, resume_checkpoint)


    def on_batch_finish(self, batch, batch_backward_count, start_imgs, ret_combine_img_org, batch_start_time, resume_checkpoint, out_dir, scaler, opts, start_epoch, generator, total_fool, batch_fool, batch_count, batch_loss):
        batch_imgs = batch_backward_count+batch_fool
        batch_asr = float(batch_fool) / float(batch_imgs)
        batch_loss = float(batch_loss) / float(batch_imgs)
        batch_count = float(batch_count) / float(batch_imgs)


        cv2.imwrite(os.path.join(out_dir, f'batch.png'), ret_combine_img_org)
        if batch % 10:
            self.save_checkpoint(total_fool, start_epoch, start_imgs, batch, generator, opts, resume_checkpoint)
        scene_time_cost = time.time() - batch_start_time
        logger.info(f"batch finish:{start_epoch} {start_imgs} fool={batch_fool} loss={batch_loss:.2f} voxel={batch_count} sec/scene={scene_time_cost:.2f}")


    def eval_and_save(self, best_eval_asr, bad_asr_count, stop_patience, generator, val_dataset, eval_idx, start_epoch, start_imgs, out_dir):
        eval_asr, savede_img = self.evaluation(generator, val_dataset)
        logger.info(f'finish evaluate: asr={eval_asr}')

        eval_info = {
            "epoch": start_epoch,
            "n_img": start_imgs,
            "asr":  eval_asr
        }

        best_pkl_path = os.path.join(out_dir, f"best.pkl")
        if not Path(best_pkl_path).exists():
            snapshot_data = dict(G=generator, G_ema=generator)
            with open(best_pkl_path, 'wb') as f:
                dill.dump(snapshot_data, f)

        if  round(eval_asr, 3) > round(best_eval_asr, 3):
            out_text_path = os.path.join(out_dir, "result.txt")
            with open(out_text_path, 'a') as f:
                f.write(f"epoch={start_epoch} asr={eval_asr}\n")
            ret_pkl_path = os.path.join(out_dir, f"best-{eval_asr:.3f}-{start_imgs}.pkl")
            snapshot_data = dict(G=generator, G_ema=generator)
            with open(ret_pkl_path, 'wb') as f:
                dill.dump(snapshot_data, f)
            cv2.imwrite(os.path.join(out_dir, f'{eval_asr:.3f}-{start_imgs}.png'), savede_img)
            logger.info(f'save pkl with {ret_pkl_path}')

        if round(eval_asr, 4) > round(best_eval_asr, 4):
            best_eval_asr = eval_asr
            bad_asr_count = 0
        else:
            bad_asr_count += 1


        is_patience_limit = bad_asr_count >= stop_patience
        done = is_patience_limit
        eval_info['bad_asr_count'] = bad_asr_count
        eval_info['is_patience_limit'] = is_patience_limit
        eval_csv_file = os.path.join(out_dir, "eval.csv")
        write_header = not Path(eval_csv_file).exists()
        with open(eval_csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=eval_info.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(eval_info)

        return best_eval_asr, bad_asr_count, done

    def compute_n_voxel_car_in_roi(self, pred_occ):
        roi_mask = get_roi_mask(self.train_params.roi_width)
        roi_mask = roi_mask.to(pred_occ.device)
        voxel_softmax = torch.softmax(pred_occ, dim=1)
        voxel_softmax_max_score, voxel_softmax_max = torch.max(voxel_softmax, dim=1)
        voxel_softmax_max = voxel_softmax_max[0]


        unsuccessful = (voxel_softmax_max != 0) & (voxel_softmax_max != 11)
        voxel_softmax_max_masked = roi_mask * unsuccessful.to(roi_mask.device)
        voxel_count = torch.count_nonzero(voxel_softmax_max_masked).item()
        return voxel_count, voxel_softmax_max


    def get_grad(self, generator, scaler, loss_type):
        params = generator.synthesis.parameters()
        grad_is_none = False
        grad_has_nan = False
        grad_list = []
        for param in params:
            grad = param.grad
            if grad is None:
                grad_is_none = True
                break
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                grad_has_nan = True
                break
            grad_list.append(grad.detach().clone())

        if grad_is_none:
            logger.error(f'loss {loss_type} grad is None')
            return None

        if grad_has_nan:
            cur_scale = scaler.get_scale()
            new_scale = cur_scale * 0.5
            scaler.update(new_scale)
            logger.error(f'loss {loss_type} grad is nan or inf new_scale={new_scale}')
            return None

        return grad_list


    def do_train(self, start_imgs, val_dataset, data_loader, start_idx_map, end_idx_map, scene_list, scene_idxs_map, generator,out_dir, checkpoint_np, resume_checkpoint, att_weight, pc_grad=False, stop_patience=2, max_epoch=10):
        generator.train()
        for param in generator.synthesis.parameters():
            param.requires_grad_(True)
        opts = torch.optim.Adam(generator.synthesis.parameters(), lr=self.train_params.lr, betas=(self.train_params.beta1, 0.999), amsgrad=self.train_params.amsgrad)
        if checkpoint_np is not None:
            opts.load_state_dict(checkpoint_np['optimizer_state_dict'])
            if 'epoch' in checkpoint_np:
                start_epoch = checkpoint_np['epoch']
            else:
                start_epoch = 0
            batch = checkpoint_np['batch']
        else:
            start_epoch = 0
            batch = 0
        logger.info(out_dir)
        done = False
        total_img = 0
        for scene in scene_list:
            idxs = scene_idxs_map[scene]
            total_img += len(idxs)
        scaler = torch.cuda.amp.GradScaler(init_scale=4096)
        total_scene = len(scene_list)
        if self.train_params.w_gen_mode == WGenMode.once:
            if self.train_params.w_tensor is not None:
                w = self.train_params.w_tensor
                with torch.no_grad():
                    cur_fake = generator.synthesis(w)
            else:
                w, cur_fake = self.get_w(generator)
        best_loss = float('inf')
        best_voxel_count = float('inf')
        best_asr = 0
        bad_asr_count = 0
        saved_imgs = 0
        batch_start_time = time.time()
        best_eval_asr = 0.0
        batch_backward_count = 0
        eval_idx = 0
        total_point_loss = 0
        total_att_loss = 0
        total_loss = 0
        total_vexel_count = 0
        total_fool = 0
        while not done:
            epoch_start_time = time.time()
            if self.train_params.w_gen_mode == WGenMode.epoch:
                w, cur_fake = self.get_w(generator)
            batch_loss = 0
            batch_count = 0
            batch_fool = 0
            batch_point_loss = 0
            batch_att_loss = 0
            batch_total_loss = 0
            batch_diver_loss = 0
            epoch_fool = 0
            epoch_loss = 0
            epoch_voxel_count = 0
            epoch_point_loss = 0
            epoch_att_loss = 0
            epoch_diver_loss = 0
            epoch_total_loss = 0
            freq_total_loss = 0
            freq_diver_loss = 0
            freq_adv_loss = 0
            freq_fool = 0
            freq_count = 0
            sum_of_grad = []
            for param_i in generator.synthesis.parameters():
                sum_of_grad.append(torch.zeros_like(param_i).to(torch.float32))
            for batch_idx, data_batch in enumerate(data_loader):
                if self.train_params.w_gen_mode == WGenMode.img:
                    w, cur_fake = self.get_w(generator)
                elif self.train_params.w_gen_mode == WGenMode.batch:
                    if (batch_idx) % self.train_params.batch_size == 0:
                        w, cur_fake = self.get_w(generator)
                if batch_idx in start_idx_map:
                    scene_info = start_idx_map[batch_idx]
                    scene_imgs = scene_info['count']
                    scene_idx = scene_info['idx']
                    scene_start_time = time.time()
                    if self.train_params.w_gen_mode == WGenMode.scene_start:
                        w, cur_fake = self.get_w(generator)
                img_metas = data_batch['img_metas'].data[0]
                bg_img = data_batch['img'].data[0].to(device0)
                bg_img_org = bg_img.detach().clone()
                mask = data_batch['mask'].data[0].to(self.device)
                car_info = data_batch['car_info']
                offset = img_metas[0]['offset']
                lidar2img = img_metas[0]['lidar2img']
                ref_mask = data_batch['ref_mask'][0][0][0]
                car_info = [cb[0] for cb in car_info]
                alpha, beta = generate_light_aug_params()
                if pc_grad:
                    opts.zero_grad()
                with torch.cuda.amp.autocast():
                    fake_img = generator.synthesis(w)
                    combine_img, ret_combine_img_org, imgs, car_mask, screen_mask =  self.fake_to_model_input(fake_img, car_info, ref_mask, bg_img, use_light_aug=self.train_params.use_light_aug, alpha=alpha, beta=beta)
                    outputs = self.model_occ(combine_img, img_metas)
                    pred, sampling_locations_list, attention_weights_logit_list, volume_size_list, indexes_list, volume_mask = outputs['pred_occ'],  outputs['sampling_locations_list'],  outputs['attention_weights_logit_list'], outputs['volume_size_list'], outputs['indexes_list'], outputs['volume_mask_list']
                    voxel_count, voxel_softmax_max = self.compute_n_voxel_car_in_roi(pred)
                    is_suc_cur_frame = (voxel_count == 0)
                    if self.att_loss_type == AttentionLossType.diver:
                        loss_diver = diver_loss_with_model(outputs, beta=1.0)
                        total_loss_tensor = loss_diver
                        loss_adv = 0
                        loss_adv_val = 0
                        loss_diver_val = loss_diver.item()
                    elif self.att_loss_type == AttentionLossType.adv_diver:
                        loss_diver = diver_loss_with_model(outputs, beta=1.0)
                        loss_adv, voxel_count, voxel_softmax_max = loss_with_roi(self.loss_type, pred, width=self.train_params.roi_width, a1=self.train_params.a1, a2=self.train_params.a2)
                        total_loss_tensor = loss_diver + loss_adv
                        loss_diver_val = loss_diver.item()
                        loss_adv_val = loss_adv.item()
                        if loss_adv == 0:
                            is_suc_cur_frame = True
                    else:
                        loss_adv, voxel_count, voxel_softmax_max = loss_with_roi(self.loss_type, pred, width=self.train_params.roi_width, a1=self.train_params.a1, a2=self.train_params.a2)
                        loss_diver_val = 0
                        if loss_adv == 0:
                            is_suc_cur_frame = True
                    if saved_imgs < 40:
                        self.save_image(out_dir, batch_idx, imgs, ret_combine_img_org, voxel_softmax_max)
                        saved_imgs += 1

                    n_imgs = start_imgs % len(data_loader)
                if is_suc_cur_frame:
                    logger.info(f'loss 2nd is zero {loss_adv} {voxel_count}')
                    total_fool += 1
                    batch_fool += 1
                    epoch_fool += 1

                if pc_grad:
                    batch_count += voxel_count
                    if not torch.isfinite(loss_adv):
                        print(f"Loss is NaN or Inf! Loss value: {loss_adv.item()}")

                        return None
                    scaler.scale(loss_adv).backward()
                    adv_grad = self.get_grad(generator, scaler, "adv")
                    if (adv_grad is not None) and self.att_loss_type in [AttentionLossType.adv_roi_point_enhance_att_patch, AttentionLossType.adv_roi_point_in_patch]:
                        opts.zero_grad()
                        generator.zero_grad()
                        bg_img_clone = bg_img_org.detach().clone()
                        with torch.cuda.amp.autocast():
                            fake_img = generator.synthesis(w)
                            combine_img, ret_combine_img_org, imgs, car_mask, screen_mask =  self.fake_to_model_input(fake_img, car_info, ref_mask, bg_img_clone, use_light_aug=self.train_params.use_light_aug, alpha=alpha, beta=beta)
                            outputs = self.model_occ(combine_img, img_metas)
                            pred, sampling_locations_list, attention_weights_logit_list, volume_size_list, indexes_list, volume_mask = outputs['pred_occ'],  outputs['sampling_locations_list'],  outputs['attention_weights_logit_list'], outputs['volume_size_list'], outputs['indexes_list'], outputs['volume_mask_list']
                        loss_point = roi_point_in_loss_func(screen_mask, sampling_locations_list, volume_size_list, volume_mask, indexes_list, only_out_patch=True)
                        scaler.scale(loss_point).backward()
                        point_grad = self.get_grad(generator, scaler, "point")
                        del outputs
                        gc.collect()
                    else:
                        point_grad = None
                        loss_point = 0
                    if (adv_grad is not None) and (self.att_loss_type ==  AttentionLossType.adv_roi_point_enhance_att_patch):
                        opts.zero_grad()
                        generator.zero_grad()
                        generator.synthesis.zero_grad()
                        bg_img_clone = bg_img_org.detach().clone()
                        with torch.cuda.amp.autocast():
                            fake_img = generator.synthesis(w)
                            combine_img, ret_combine_img_org, imgs, car_mask, screen_mask =  self.fake_to_model_input(fake_img, car_info, ref_mask, bg_img_clone, use_light_aug=self.train_params.use_light_aug, alpha=alpha, beta=beta)
                            outputs = self.model_occ(combine_img, img_metas)
                            pred, sampling_locations_list, attention_weights_logit_list, volume_size_list, indexes_list, volume_mask = outputs['pred_occ'],  outputs['sampling_locations_list'],  outputs['attention_weights_logit_list'], outputs['volume_size_list'], outputs['indexes_list'], outputs['volume_mask_list']
                        loss_att = roi_attention_loss_func(screen_mask,  sampling_locations_list, attention_weights_logit_list, volume_size_list,volume_mask, indexes_list, is_enhance=True)
                        scaler.scale(loss_att).backward()
                        att_grad = self.get_grad(generator, scaler, "att")
                        del outputs
                        gc.collect()
                    else:
                        att_grad = None
                        loss_att = 0
                    final_grad, info = compute_pc_grad(adv_grad, point_grad, att_grad, att_weight)
                    if final_grad is not None:
                        info['epoch'] = start_epoch
                        info['n_img'] = start_imgs
                        loss_csv_path = os.path.join(out_dir, f'pcgrad-prg.csv')
                        exit_file = os.path.exists(loss_csv_path)
                        with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.DictWriter(f, fieldnames=info.keys())
                            if not exit_file:
                                writer.writeheader()
                            writer.writerow(info)
                        for i, grad_i in enumerate(final_grad):
                            sum_of_grad[i] += grad_i
                        batch_backward_count += 1
                    total_loss_val = 0
                    loss_adv_val = loss_adv.item() if loss_adv else 0
                    loss_point_val = loss_point.item() if loss_point else 0
                    loss_att_val = loss_att.item() if loss_att else 0
                    del outputs, adv_grad, point_grad, att_grad
                    gc.collect()
                else:
                    if self.att_loss_type == AttentionLossType.diver or self.att_loss_type == AttentionLossType.adv_diver:
                        loss_point_val = 0
                        loss_att_val = 0
                    else:
                        loss_adv_val = loss_adv.item() if loss_adv else 0
                        total_loss_tensor = loss_adv
                        if self.att_loss_type in [AttentionLossType.adv_roi_point_enhance_att_patch, AttentionLossType.adv_roi_point_in_patch]:
                            loss_point = roi_point_in_loss_func(screen_mask, sampling_locations_list, volume_size_list, volume_mask, indexes_list, only_out_patch=True)
                            total_loss_tensor += att_weight * loss_point
                            loss_point_val = loss_point.item()
                        else:
                            loss_point_val = 0
                        if self.att_loss_type in [AttentionLossType.adv_roi_point_enhance_att_patch]:
                            loss_att = roi_attention_loss_func(screen_mask,  sampling_locations_list, attention_weights_logit_list, volume_size_list,volume_mask, indexes_list, is_enhance=True)
                            total_loss_tensor += att_weight * loss_att
                            loss_att_val = loss_att.item()
                        else:
                            loss_att_val = 0
                    scaler.scale(total_loss_tensor).backward()
                    total_loss_val = total_loss_tensor.item()
                    batch_backward_count += 1

                batch_total_loss += total_loss_val
                freq_total_loss += total_loss_val
                batch_diver_loss += loss_diver_val
                freq_diver_loss += loss_diver_val
                freq_adv_loss += loss_adv_val
                batch_point_loss += loss_point_val
                batch_att_loss += loss_att_val
                batch_loss += loss_adv_val
                logger.info(f'{start_epoch} {n_imgs}/{len(data_loader)}: loss_total:{total_loss_val} loss_adv: {loss_adv_val}, loss_diver:{loss_diver_val}, loss_point={loss_point_val}, loss_att={loss_att_val}, vexel_count: {voxel_count}')


                if batch_backward_count == self.train_params.batch_size:
                    if pc_grad:
                        with torch.no_grad():
                            for i, param in enumerate(generator.synthesis.parameters()):
                                param.grad = sum_of_grad[i]


                    scaler.step(opts)


                    scaler.update()
                    opts.zero_grad()
                    if pc_grad:
                        for i in range(len(sum_of_grad)):
                            sum_of_grad[i].zero_()
                    epoch_fool += batch_fool
                    epoch_loss += batch_loss
                    epoch_voxel_count += batch_count
                    epoch_point_loss += batch_point_loss
                    epoch_att_loss += batch_att_loss
                    epoch_diver_loss += batch_diver_loss
                    epoch_total_loss += batch_total_loss
                    n_imgs = start_imgs % len(data_loader)

                    fool_rate = batch_fool/float(batch_backward_count)
                    avg_batch_loss_adv = float(batch_loss) / float(batch_backward_count)
                    avg_batch_loss_point = float(batch_point_loss) / float(batch_backward_count)
                    avg_batch_loss_att = float(batch_att_loss) / float(batch_backward_count)
                    avg_batch_loss_diver = float(epoch_diver_loss) / float(batch_backward_count)
                    avg_batch_loss_total = float(epoch_total_loss) / float(batch_backward_count)
                    info = {'epoch': start_epoch, 'batch': batch, 'n_img': n_imgs, 'total_loss': avg_batch_loss_total, 'diver_loss': avg_batch_loss_diver, 'adv_loss': avg_batch_loss_adv, 'point_loss': avg_batch_loss_point, 'att_loss': avg_batch_loss_att, 'fool_rate': fool_rate, 'batch_count': batch_count}
                    loss_csv_path = os.path.join(out_dir, f'batch-loss.csv')
                    exit_file = os.path.exists(loss_csv_path)
                    with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=info.keys())
                        if not exit_file:
                            writer.writeheader()
                        writer.writerow(info)
                    self.on_batch_finish(batch, batch_backward_count, n_imgs, ret_combine_img_org, batch_start_time, resume_checkpoint, out_dir, scaler, opts, start_epoch, generator, total_fool, batch_fool, batch_count, batch_loss)
                    batch_fool = 0
                    batch_count = 0
                    batch_loss = 0
                    batch_point_loss = 0
                    batch_att_loss = 0
                    batch_backward_count = 0
                    batch_total_loss = 0
                    batch_start_time = time.time()
                    batch += 1

                freq_count += 1
                if self.train_params.eval_batch > 0:
                    info = {'epoch': start_epoch, 'batch': batch, 'n_img': n_imgs, 'total_loss': freq_total_loss/freq_count, 'diver_loss': freq_diver_loss/freq_count, 'adv_loss': freq_adv_loss/freq_count}
                    loss_csv_path = os.path.join(out_dir, f'freq-loss.csv')
                    exit_file = os.path.exists(loss_csv_path)
                    with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=info.keys())
                        if not exit_file:
                            writer.writeheader()
                        writer.writerow(info)
                    freq_count = 0
                    freq_total_loss = 0
                    freq_diver_loss = 0
                    freq_adv_loss = 0


                    if batch > 0 and batch % self.train_params.eval_batch == 0:
                        eval_asr, bad_asr_count_org, is_done = self.eval_and_save(best_eval_asr, bad_asr_count, stop_patience, generator, val_dataset, eval_idx, start_epoch, start_imgs, out_dir)
                        eval_idx += 1
                        best_eval_asr = eval_asr
                        bad_asr_count = bad_asr_count_org
                        if is_done:
                            done = True
                            break
                start_imgs += 1

            total_vexel_count += epoch_voxel_count
            total_loss += epoch_loss
            total_point_loss += epoch_point_loss
            total_att_loss += epoch_att_loss
            start_epoch += 1
            if start_epoch >= max_epoch:
                done = True
            total_fool_rate = float(total_fool) /  float(total_img)
            time_epoch = time.time() - epoch_start_time
            gpu_use = torch.cuda.max_memory_allocated() / 1e9
            if total_loss < best_loss:
                best_loss = total_loss
            if total_vexel_count < best_voxel_count:
                best_voxel_count = total_vexel_count

            logger.info(f'Finish epoch! {start_epoch} fool_rate:{total_fool_rate:.2f} fool={total_fool}/{total_img} loss={total_loss:.4f}{best_loss:.4f} voxel_count={total_vexel_count}/{best_voxel_count} sec/epoch={time_epoch} gpu_use={gpu_use:.2f}')


        ret_pkl_path = os.path.join(out_dir, f"success.pkl")
        snapshot_data = dict(G=generator, G_ema=generator)
        with open(ret_pkl_path, 'wb') as f:
            dill.dump(snapshot_data, f)
        logger.info(f"Training completed successfully ! epochs={start_epoch}")


    def evaluation(self, generator, val_dataset):
        fooled = 0
        detected_total = 0
        ret_combine_img = None
        w, cur_fake = self.get_w(generator)
        cur_fake = generator.synthesis(w)
        for batch_idx, data_batch in enumerate(val_dataset):
            img_metas = data_batch['img_metas'].data[0]
            bg_img = data_batch['img'].data[0].to(self.device)
            car_info = data_batch['car_info']
            ref_mask = data_batch['ref_mask'][0][0][0]
            car_info = [cb[0] for cb in car_info]
            with torch.inference_mode():
                with torch.cuda.amp.autocast():

                    bg_clone = bg_img.detach().clone()
                    benign_img, _, _ = prepare_with_fake_img(use_light_aug=False, car_info=car_info, ref_mask_tensor=ref_mask, fimg_list=None, bg_img=bg_clone, is_bgr=False, n_gaps=self.train_params.n_gaps, max_offset=self.train_params.max_offset)
                    mean = torch.tensor(self.mean).reshape(1, 1, 3, 1, 1).to(bg_img.device)
                    std = torch.tensor(self.std).reshape(1, 1, 3, 1, 1).to(bg_img.device)
                    benign_nom = (benign_img - mean) / std
                    benign_combine = pad_to_multiple_tensor(benign_nom, self.size_divisor, self.pad_val)
                    benign_outputs = self.model_occ(benign_combine, img_metas)
                    benign_pred = benign_outputs['pred_occ']
                    roi_mask = get_roi_mask(self.train_params.roi_width).to(bg_img.device)
                    benign_argmax = benign_pred[0].argmax(dim=0)
                    car_in_roi = torch.count_nonzero(roi_mask * (benign_argmax == 4)).item()
                    if car_in_roi == 0:
                        continue
                    detected_total += 1

                    combine_img, ret_combine_img, imgs, car_mask, screen_mask =  self.fake_to_model_input(cur_fake, car_info, ref_mask, bg_img)
                    outputs = self.model_occ(combine_img, img_metas)
                    pred = outputs['pred_occ']
                    voxel_count, voxel_softmax_max = self.compute_n_voxel_car_in_roi(pred)
                    if voxel_count == 0:
                        fooled += 1
        asr = fooled / detected_total if detected_total > 0 else 0.0
        print(f'eval finish. asr={asr} (detected={detected_total}/{len(val_dataset)})')
        return asr, ret_combine_img

def get_dataloader(cfg, dataset_pkl_path, h_list, d_list, r_list,shuffle=False, workers_per_gpu=0, start_idx=0,strict_uniform=False):
    test_config = cfg.data.test
    test_config.ann_file = dataset_pkl_path
    test_pipeline = [
        dict(type='LoadMultiViewImageFromFiles', to_float32=False),
        dict(type='CarImagePipeline', h_list=h_list, d_list=d_list, r_list=r_list, size_divisor=32, device=device0,strict_uniform=strict_uniform, **cfg.img_norm_cfg),
        dict(type='CustomCollect3D', keys=['img', 'gt_occ', 'car_info', 'ref_mask', 'mask', 'lidar2img'])
    ]
    test_config.pipeline = test_pipeline


    dataset = build_dataset(cfg.data.test)
    if start_idx != 0:
        indices = [(start_idx + i) % len(dataset) for i in range(len(dataset))]
        subdataset = torch.utils.data.Subset(dataset, indices)
        dataset = subdataset
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=workers_per_gpu, dist=1, shuffle=shuffle)
    return data_loader

def clean_and_convert_dict(input_dict):

    def process_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            else:
                return None
        if isinstance(value, Enum):
            return value.name
        return value

    return {k: v for k, v in ((k, process_value(v)) for k, v in input_dict.items()) if v is not None}

def get_param_dict(attack_param):
    params = copy.deepcopy(attack_param)
    params = clean_and_convert_dict(params)
    if 'img_metas' in params:
        params.pop('img_metas')
    return params

def get_dir_str(opts):

    def list_str(lst):
        if len(lst) == 1:
            return lst[0]
        return ",".join(str(list_item) for list_item in lst)

    dir_str = f"d-{list_str(opts.d_list)}-h-{list_str(opts.h_list)}-r-{list_str(opts.r_list)}"
    if opts.batch_size != 1:
        dir_str += f"-batch_{opts.batch_size}"
    dir_str += f"-lr-{opts.lr}-beta1-{opts.beta1}"
    if opts.ld_factor != 0:
        dir_str += f"-l_{opts.per_loss.name}_{opts.ld_factor}"
    if opts.use_light_aug == True:
        dir_str += f"-light_aug"
    if opts.n_gaps > 0:
        dir_str += f"-gap_{opts.max_offset}"
    if opts.amsgrad == True:
        dir_str += f"-amsgrad"
    else:
        dir_str += f"-no-amsgrad"
    if opts.one_bg:
        dir_str += "-one_bg"
    dir_str += f"-w_{opts.w_gen_mode.name}"
    if opts.init_w is not None:
        dir_str += "-with_w"

    return dir_str


def train_f(t_param):
    opts = t_param.args
    config_path = os.path.join(_ROOT, 'SurroundOcc', 'projects', 'configs', 'surroundocc', 'surroundocc_inference.py')
    checkpoint_path = os.path.join(_ROOT, 'data', 'weight', 'surroundocc.pth')
    cfg = Config.fromfile(config_path)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if cfg.get('cudnn_benchmark', False) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    print('finish load config')
    for loss_att in opts.loss_att:


        opts.init_type = InitType(opts.init_type)
        opts.w_gen_mode = WGenMode(opts.w_gen_mode)
        out_dir = os.path.join(_ROOT, 'output_new', 'adv_train_amp', f"{opts.model_names}", f"{loss_att.name}", get_dir_str(opts))
        os.makedirs(out_dir, exist_ok=True)
        if opts.resume_checkpoint is None:
            opts.resume_checkpoint = os.path.join(out_dir, f"checkopint.pth")

        if opts.init_w is not None and os.path.isfile(opts.init_w):
            w_np = np.load(opts.init_w)
            w_tensor = torch.tensor(w_np, dtype=torch.float32, device=device0)
            opts.w_tensor = w_tensor
        else:
            opts.w_tensor = None

        resume_checkpoint = opts.resume_checkpoint
        generator, D_m = load_generator_styleganxl(opts.pkl)
        c_dim = generator.c_dim
        if resume_checkpoint is None:
            resume_checkpoint = os.path.join(out_dir, f"checkopint.pth")
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Loading checkpoint from {resume_checkpoint}")
            checkpoint_np = torch.load(resume_checkpoint)
            start_images = checkpoint_np['total_images']
            print(f"Resuming training, total images trained: {start_images}")
        else:
            start_images = 0
            checkpoint_np = None
        print('befor get data loader')

        val_dataset = get_dataloader(cfg, strict_uniform=opts.strict_uniform, dataset_pkl_path=opts.eval_pkl, shuffle=False, h_list=opts.h_list, d_list=opts.d_list, r_list=opts.r_list)
        model, cfg = prepare_with_cfg(config_path, checkpoint_path, device0)
        model = model.to(device0)
        model.eval()
        if loss_att == AttentionLossType.diver or loss_att == AttentionLossType.adv_diver:
            enable_hook = True
        else:
            enable_hook = False
        model_occ = SurroundOccModel(model, enable_hook=enable_hook)
        trainer = GanAdvTrainer(model_occ, device0, c_dim, out_dir, opts.loss_type[0], loss_att, cfg=opts)
        opts_copy = copy.deepcopy(opts)
        opts_copy.loss_type = opts.loss_type[0].name
        opts_copy.loss_att = loss_att.name
        save_param = get_param_dict(opts_copy)
        with open(os.path.join(out_dir, 'training_options.json'), 'w') as f:
            json.dump(save_param, f)
        generator = generator.to(device0)
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Loading checkpoint from {resume_checkpoint}")
            checkpoint_np = torch.load(resume_checkpoint)
            start_imgs = checkpoint_np['total_images']
            print(f'resum with {start_imgs}')
            generator.load_state_dict(checkpoint_np['model_state_dict'])
        else:
            checkpoint_np = None
            start_imgs = 0
        data_loader = get_dataloader(cfg, start_idx=start_imgs, shuffle=opts.shuffle, dataset_pkl_path=opts.dataset_pkl, h_list=opts.h_list, d_list=opts.d_list, r_list=opts.r_list, workers_per_gpu=opts.workers_per_gpu)
        scene_list, scene_idxs_map, original_dataset, start_idx_map, end_idx_map = get_scene_list(data_loader)
        trainer.do_train(start_imgs, val_dataset, data_loader, start_idx_map, end_idx_map, scene_list, scene_idxs_map, generator,out_dir, checkpoint_np, resume_checkpoint, opts.att_weight, opts.pc_grad, stop_patience=opts.stop_patience, max_epoch=opts.max_epoch)


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    ret_list =  s.split(',')


    processed_list = []
    for item in ret_list:
        item = item.strip()  
        try:
            processed_list.append(float(item))
        except ValueError:
            processed_list.append(item)

    return processed_list

def parse_loss_type_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    ret_list =  s.split(',')


    processed_list = []
    for item in ret_list:
        item = item.strip()
        try:
            processed_list.append(LossType(item))
        except ValueError:
            processed_list.append(item)

    return processed_list

def parse_loss_att_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    ret_list =  s.split(',')


    processed_list = []
    for item in ret_list:
        item = item.strip()
        processed_list.append(AttentionLossType(item))

    return processed_list

@logger.catch
@click.command()
@click.option('--pkl',        type=str, required=True, help='Path to the network pickle files')
@click.option('--dataset_pkl',        type=str, required=True, help='Path to the dataset pickle files')
@click.option('--model_names', type=str, required=True, help='Model names to be used for training')
@click.option('--num_fake_img', type=int, default=11, help='Number of fake images to be generated (paper: 11 segments)')
@click.option('--lr', type=float, default=0.002, help='Learning rate for training (paper: 0.002)')
@click.option('--amsgrad', type=bool, default=False, help='Use AMSGrad for Adam optimizer (paper: standard Adam, amsgrad=False)')
@click.option('--loss_type', type=parse_loss_type_separated_list, default='logit_softplus_paper', help='Loss type for training (paper Eq.5-8)')
@click.option('--loss_att', type=parse_loss_att_separated_list, default='adv_diver', help='Loss type for training (paper Eq.9: diver+occ)')
@click.option('--init_type', type=str, default='zero', help='Initialization type for training')
@click.option('--init_w', type=str, default=None, help='Path to the initial weights for training')
@click.option('--resume_checkpoint', type=str, default=None, help="path resum checckpoint in")
@click.option('--with_scale', type=bool, default=True)
@click.option('--ld_factor', type=float, default=0.1)
@click.option('--seed',       type=int, default=47)
@click.option('--per_loss', type=PerLossType, default='d0_l1')
@click.option('--epsilon', type=float, default=0.03)
@click.option('--sparsity', type=float, default=0.1)
@click.option('--beta1', type=float, default=0.9)
@click.option('--use_light_aug', type=bool, default=True, help='Apply photometric augmentation (paper Eq.10)')
@click.option('--n_layers', type=int, default=11)
@click.option('--max_h', type=float, default=20, help='HSV hue constraint Hmax (paper: 20)')
@click.option('--n_gaps', type=float, default=0, help='Whether to use random offset for training')
@click.option('--max_offset', type=float, default=0, help='Maximum offset for random offset augmentation')
@click.option('--f_max', default=8, type=int)
@click.option('--h_list', default=[1.5], type=parse_comma_separated_list)
@click.option('--d_list', default=[0.4], type=parse_comma_separated_list)
@click.option('--r_list', default=[0.0], type=parse_comma_separated_list)
@click.option('--workers_per_gpu', default=0, type=int)
@click.option('--batch_size', default=16, type=int, help='Frames accumulated per Adam step (paper: 16)')
@click.option('--one_bg', default=False, type=bool)
@click.option('--w_gen_mode', default="img", type=str, help='Per-frame z resampling (paper Algorithm 1)')
@click.option('--eval_pkl',        type=str, help='Path to the dataset pickle files')
@click.option('--stop_patience', '--stop_pation', type=int, default=2, help='Early stopping patience (paper: 2 consecutive epochs without ASR improvement)')
@click.option('--max_epoch', type=int, default=10, help='Maximum training epochs (paper: 10)')
@click.option('--shuffle', type=bool, default=False, help="if shuffle the training dataset or not")
@click.option('--eval_batch', type=int, default=104, help='Validate ASR every N Adam-step batches (paper Algorithm 1: n_batch mod 104 = 0)')
@click.option('--roi_width', type=float, default=2.0)
@click.option('--use_detect', type=bool, default=False)
@click.option('--a1', type=float, default=1.0)
@click.option('--a2', type=float, default=0.8)
@click.option('--eq_class', default=False, type=bool)
@click.option('--strict_uniform', default=False, type=bool)
@click.option('--pc_grad', default=False, type=bool)
@click.option('--att_weight', type=float, default=0.5)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    t_param = TrainParam(opts)
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    train_f(t_param)


if __name__ == '__main__':
    main()