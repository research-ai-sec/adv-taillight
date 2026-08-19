"""Standalone ASR evaluation for the adversarial taillight attack.

Given a trained StyleGAN-XL generator and a nuScenes validation split, this
script composites the adversarial taillight appearance onto the target car and
measures the attack success rate (ASR) against the SurroundOcc victim occupancy
network.

ASR = #frames where the target car is no longer predicted occupied in the RoI
      / #frames where the car was originally detected by the victim.

Usage:
    python metric.py \
        --pkl         /path/to/stylegan-xl-imagenet32.pkl   # base generator
        --ckp         /path/to/best.pkl                     # fine-tuned attack generator
        --dataset_pkl /path/to/nuscenes_infos_val.pkl      # validation scene pickle
        --name        eval_v1
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
from types import SimpleNamespace

import click
import dill
import numpy as np
import torch
from mmdet3d.datasets import build_dataset
from SurroundOcc.projects.mmdet3d_plugin.datasets.builder import build_dataloader

from attack.styleganxl import get_w_from_seed, load_generator_styleganxl
from detector.surround_occ import (
    pad_to_multiple_tensor,
    prepare_with_cfg,
    prepare_with_fake_img,
)
from detector.voxel_car_detector import get_roi_mask
from spocc.surround_occ_model import SurroundOccModel
from utils.scene_utils import get_scene_list

MEAN = [103.530, 116.280, 123.675]
STD = [1.0, 1.0, 1.0]
SIZE_DIVISOR = 32
PAD_VAL = 0
CAR_CLASS_ID = 4  # nuScenes occupancy class index of "car"


def parse_float_list(_ctx, _param, value):
    """Click callback that parses a comma-separated string into a float list."""
    if value is None:
        return []
    return [float(x) for x in value.split(",")]


def get_dataloader(cfg, dataset_pkl_path, h_list, d_list, r_list, workers_per_gpu=0, device="cuda"):
    """Build the validation dataloader with the CarImagePipeline (same as training)."""
    test_config = cfg.data.test
    test_config.ann_file = dataset_pkl_path
    test_pipeline = [
        dict(type="LoadMultiViewImageFromFiles", to_float32=False),
        dict(
            type="CarImagePipeline",
            h_list=h_list,
            d_list=d_list,
            r_list=r_list,
            size_divisor=SIZE_DIVISOR,
            device=device,
            strict_uniform=False,
            **cfg.img_norm_cfg,
        ),
        dict(
            type="CustomCollect3D",
            keys=["img", "gt_occ", "car_info", "ref_mask", "mask", "lidar2img"],
        ),
    ]
    test_config.pipeline = test_pipeline
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        dist=1,
        shuffle=False,
    )
    return data_loader


def generate_attack_images(generator, num_fake_img, device):
    """Sample a latent and synthesize the adversarial taillight appearance (Eq. 2-10)."""
    z = torch.randn(size=(num_fake_img, generator.z_dim), device=device)
    if generator.c_dim != 0:
        class_idx = torch.randint(0, generator.c_dim, size=(num_fake_img,)).to(device).to(torch.long)
    else:
        class_idx = None
    w = get_w_from_seed(generator, z, num_fake_img, device=device, class_idx=class_idx)
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            cur_fake = generator.synthesis(w)
    return cur_fake


def h_b_clip_func(rgb, max_h=20):
    """Photometric hue constraint: keep only red hues up to H_max (paper, Eq. 10)."""
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


def count_car_voxels_in_roi(pred_occ, roi_width, device):
    """Count non-background voxels inside the RoI after the attack (0 == fully fooled)."""
    roi_mask = get_roi_mask(roi_width).to(device)
    voxel_softmax = torch.softmax(pred_occ, dim=1)
    _, voxel_softmax_max = torch.max(voxel_softmax, dim=1)
    voxel_softmax_max = voxel_softmax_max[0]

    unsuccessful = (voxel_softmax_max != 0) & (voxel_softmax_max != 11)
    voxel_softmax_max_masked = roi_mask * unsuccessful.to(roi_mask.device)
    voxel_count = torch.count_nonzero(voxel_softmax_max_masked).item()
    return voxel_count


def load_finetuned_generator(generator, ckp_path):
    """Apply a fine-tuned attack checkpoint onto the base generator.

    Accepts either a torch-saved resume checkpoint (with a "model_state_dict" key)
    or a dill-saved generator snapshot (dict(G=..., G_ema=...)) produced by
    ``gan_adv_train.py``.
    """
    try:
        ckpt = torch.load(ckp_path)
        if "model_state_dict" in ckpt:
            generator.load_state_dict(ckpt["model_state_dict"])
            return
    except Exception:
        pass
    with open(ckp_path, "rb") as f:
        snapshot = dill.load(f)
    generator.load_state_dict(snapshot["G"].state_dict())


def evaluate(model_occ, cur_fake, val_loader, opts, device):
    """Compute per-scene and overall ASR over the validation scenes."""
    scene_list, scene_idxs_map, original_dataset = get_scene_list(val_loader)

    mean = torch.tensor(MEAN).reshape(1, 1, 3, 1, 1).to(device)
    std = torch.tensor(STD).reshape(1, 1, 3, 1, 1).to(device)

    n_detected = 0
    n_fooled = 0
    scene_summary = []
    for scene in scene_list:
        idxs = scene_idxs_map[scene]
        detected_scene = 0
        fooled_scene = 0
        for idx in idxs:
            data_batch = original_dataset[idx]
            img_metas = data_batch["img_metas"].data[0]
            bg_img = data_batch["img"].data[0].to(device)
            car_info = [cb[0] for cb in data_batch["car_info"]]
            ref_mask = data_batch["ref_mask"][0][0][0]

            with torch.inference_mode():
                with torch.cuda.amp.autocast():
                    # Benign reference: only count frames the victim originally detects.
                    bg_clone = bg_img.detach().clone()
                    benign_img, _, _ = prepare_with_fake_img(
                        use_light_aug=False,
                        car_info=car_info,
                        ref_mask_tensor=ref_mask,
                        fimg_list=None,
                        bg_img=bg_clone,
                        is_bgr=False,
                        n_gaps=opts.n_gaps,
                        max_offset=opts.max_offset,
                    )
                    benign_nom = (benign_img - mean) / std
                    benign_combine = pad_to_multiple_tensor(benign_nom, SIZE_DIVISOR, PAD_VAL)
                    benign_outputs = model_occ(benign_combine, img_metas)
                    benign_pred = benign_outputs["pred_occ"]
                    roi_mask = get_roi_mask(opts.roi_width).to(bg_img.device)
                    benign_argmax = benign_pred[0].argmax(dim=0)
                    car_in_roi = torch.count_nonzero(roi_mask * (benign_argmax == CAR_CLASS_ID)).item()
                    if car_in_roi == 0:
                        continue
                    detected_scene += 1

                    # Attacked frame: adversarial taillight composited on the car.
                    imgs = (cur_fake * 127.5 + 128).clamp(0, 255).to(torch.float32)
                    imgs = h_b_clip_func(imgs, opts.max_h)
                    result_img, _, _ = prepare_with_fake_img(
                        use_light_aug=False,
                        car_info=car_info,
                        ref_mask_tensor=ref_mask,
                        fimg_list=imgs,
                        bg_img=bg_img,
                        is_bgr=False,
                        n_gaps=opts.n_gaps,
                        max_offset=opts.max_offset,
                    )
                    result_nom = (result_img - mean) / std
                    combine_img = pad_to_multiple_tensor(result_nom, SIZE_DIVISOR, PAD_VAL)
                    outputs = model_occ(combine_img, img_metas)
                    voxel_count = count_car_voxels_in_roi(outputs["pred_occ"], opts.roi_width, device)
                    if voxel_count == 0:
                        fooled_scene += 1

        scene_asr = float(fooled_scene) / float(detected_scene) if detected_scene > 0 else 0.0
        n_detected += detected_scene
        n_fooled += fooled_scene
        scene_summary.append(
            {
                "scene": scene,
                "detected": detected_scene,
                "fooled": fooled_scene,
                "asr": scene_asr,
            }
        )
        print(f"scene {scene}: fooled {fooled_scene}/{detected_scene} asr={scene_asr:.3f}")

    asr = float(n_fooled) / float(n_detected) if n_detected > 0 else 0.0
    print(f"eval finish. asr={asr:.3f} (detected={n_detected})")
    return {
        "asr": asr,
        "n_detected": n_detected,
        "n_fooled": n_fooled,
        "scenes": scene_summary,
    }


@click.command()
@click.option("--pkl", type=str, required=True, help="Base StyleGAN-XL generator pickle (ImageNet-32)")
@click.option("--ckp", type=str, default=None, help="Fine-tuned attack generator checkpoint")
@click.option("--dataset_pkl", type=str, required=True, help="Validation dataset pickle (nuScenes infos)")
@click.option("--name", type=str, required=True, help="Experiment name (output sub-directory)")
@click.option("--num_fake_img", type=int, default=11, help="Number of taillight appearance segments N (paper: 11)")
@click.option("--max_h", type=float, default=20, help="HSV hue constraint H_max (paper: 20)")
@click.option("--roi_width", type=float, default=2.0, help="RoI half-width in meters (paper: 2.0)")
@click.option("--n_gaps", type=float, default=0, help="Random-placement mode (paper: off)")
@click.option("--max_offset", type=float, default=0, help="Max random placement offset (paper: 0)")
@click.option("--h_list", default="1.5", type=str, callback=parse_float_list, help="Comma-separated target-car heights")
@click.option("--d_list", default="0.4", type=str, callback=parse_float_list, help="Comma-separated target-car distances")
@click.option("--r_list", default="0.0", type=str, callback=parse_float_list, help="Comma-separated target-car yaw offsets")
@click.option("--seed", type=int, default=47)
@click.option("--workers_per_gpu", type=int, default=0)
def main(**kwargs):
    opts = SimpleNamespace(**kwargs)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = os.path.join(
        _ROOT, "SurroundOcc", "projects", "configs", "surroundocc", "surroundocc_inference.py"
    )
    checkpoint_path = os.path.join(_ROOT, "data", "weight", "surroundocc.pth")

    model, cfg = prepare_with_cfg(config_path, checkpoint_path, device)
    model_occ = SurroundOccModel(model, enable_hook=False).to(device)
    model_occ.eval()

    val_loader = get_dataloader(
        cfg,
        opts.dataset_pkl,
        opts.h_list,
        opts.d_list,
        opts.r_list,
        workers_per_gpu=opts.workers_per_gpu,
        device=device,
    )

    generator = load_generator_styleganxl(opts.pkl)
    generator = generator.to(device)
    generator.eval()
    if opts.ckp is not None and os.path.exists(opts.ckp):
        load_finetuned_generator(generator, opts.ckp)

    cur_fake = generate_attack_images(generator, opts.num_fake_img, device)
    summary = evaluate(model_occ, cur_fake, val_loader, opts, device)

    out_dir = os.path.join(_ROOT, "output", "metric", opts.name)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved result to", result_path)


if __name__ == "__main__":
    main()
