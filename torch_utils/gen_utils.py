

import os
import re
import json

from typing import List, Tuple, Union, Optional
from collections import OrderedDict
from locale import atof

import click
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib





def create_image_grid(images: np.ndarray, grid_size: Optional[Tuple[int, int]] = None):


    assert images.ndim == 3 or images.ndim == 4, f'Images has {images.ndim} dimensions (shape: {images.shape})!'
    num, img_h, img_w, c = images.shape

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)

        if grid_w is None:
            grid_w = num // grid_h + min(num % grid_h, 1)
        elif grid_h is None:
            grid_h = num // grid_w + min(num % grid_w, 1)


    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)


    assert grid_w * grid_h >= num, 'Number of rows and columns must be greater than the number of images!'

    grid = np.zeros([grid_h * img_h, grid_w * img_h] + list(images.shape[-1:]), dtype=images.dtype)

    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid





def parse_fps(fps: Union[str, int]) -> int:

    if isinstance(fps, int):
        return max(fps, 1)
    try:
        fps = int(atof(fps))
        return max(fps, 1)
    except ValueError:
        print(f'Typo in "--fps={fps}", will use default value of 30')
        return 30


def num_range(s: str, remove_repeated: bool = True) -> List[int]:

    str_list = s.split(',')
    nums = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for el in str_list:
        match = range_re.match(el)
        if match:

            lower, upper = int(match.group(1)), int(match.group(2))
            if lower <= upper:
                r = list(range(lower, upper + 1))
            else:
                r = list(range(upper, lower + 1))

            nums.extend(r)
        else:

            try:
                nums.append(int(atof(el)))
            except ValueError:
                continue

    if remove_repeated:
        nums = list(OrderedDict.fromkeys(nums))
    return nums


def parse_slowdown(slowdown: Union[str, int]) -> int:


    if not isinstance(slowdown, int):
        try:
            slowdown = atof(slowdown)
        except ValueError:
            print(f'Typo in "{slowdown}"; will use default value of 1')
            slowdown = 1
    assert slowdown > 0, '"slowdown" cannot be negative or 0!'

    slowdown = 2**int(np.rint(np.log2(slowdown)))
    return max(slowdown, 1)


def parse_new_center(s: str) -> Tuple[str, Union[int, np.ndarray]]:

    try:
        new_center = int(s)
        return s, new_center
    except ValueError:
        new_center = get_w_from_file(s, return_ext=False)
        return s, new_center





def compress_video(
        original_video: Union[str, os.PathLike],
        original_video_name: Union[str, os.PathLike],
        outdir: Union[str, os.PathLike],
        ctx: click.Context) -> None:

    try:
        import ffmpeg
    except (ModuleNotFoundError, ImportError):
        ctx.fail('Missing ffmpeg! Install it via "pip install ffmpeg-python"')

    print('Compressing the video...')
    resized_video_name = os.path.join(outdir, f'{original_video_name}-compressed.mp4')
    ffmpeg.input(original_video).output(resized_video_name).run(capture_stdout=True, capture_stderr=True)
    print('Success!')





def interpolation_checks(
        t: Union[float, np.ndarray],
        v0: np.ndarray,
        v1: np.ndarray) -> Tuple[Union[float, np.ndarray], np.ndarray, np.ndarray]:


    assert np.min(t) >= 0.0 and np.max(t) <= 1.0

    if not isinstance(v0, np.ndarray):
        v0 = np.array(v0)
    if not isinstance(v1, np.ndarray):
        v1 = np.array(v1)

    assert v0.shape == v1.shape, f'Incompatible shapes! v0: {v0.shape}, v1: {v1.shape}'
    return t, v0, v1


def lerp(
        t: Union[float, np.ndarray],
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray]) -> np.ndarray:

    t, v0, v1 = interpolation_checks(t, v0, v1)
    v2 = (1.0 - t) * v0 + t * v1
    return v2


def slerp(
        t: Union[float, np.ndarray],
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray],
        dot_threshold: float = 0.9995) -> np.ndarray:

    t, v0, v1 = interpolation_checks(t, v0, v1)

    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    dot = np.sum(v0 * v1)

    if np.abs(dot) > dot_threshold:
        return lerp(t, v0, v1)

    dot = np.clip(dot, -1.0, 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


def interpolate(
        v0: Union[float, list, tuple, np.ndarray],
        v1: Union[float, list, tuple, np.ndarray],
        n_steps: int,
        interp_type: str = 'spherical',
        smooth: bool = False) -> np.ndarray:

    t_array = np.linspace(0, 1, num=n_steps, endpoint=False)

    if smooth:



        t_array = t_array ** 2 * (3 - 2 * t_array)

    funcs_dict = {'linear': lerp, 'spherical': slerp}
    vectors = np.array([funcs_dict[interp_type](t, v0, v1) for t in t_array], dtype=np.float32)
    return vectors





def double_slowdown(latents: np.ndarray, duration: float, frames: int) -> Tuple[np.ndarray, float, int]:


    z = np.empty(np.multiply(latents.shape, [2, 1, 1]), dtype=np.float32)

    for i in range(len(latents)):
        z[2 * i] = latents[i]

    for i in range(1, len(z), 2):

        z[i] = slerp(0.5, z[i - 1], z[i + 1]) if i != len(z) - 1 else slerp(0.5, z[0], z[i - 1])


    return z, 2 * duration, 2 * frames





def make_affine_transform(m: torch.Tensor = None,
                          angle: float = 0.0,
                          translate_x: float = 0.0,
                          translate_y: float = 0.0,
                          scale_x: float = 1.0,
                          scale_y: float = 1.0,
                          shear_x: float = 0.0,
                          shear_y: float = 0.0,
                          mirror_x: bool = False,
                          mirror_y: bool = False) -> np.array:


    if m is None:
        m = np.eye(3, dtype=np.float64)
    else:
        m = m.cpu().numpy()


    rotation_matrix = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                [-np.sin(angle), np.cos(angle), 0.0],
                                [0.0, 0.0, 1.0]], dtype=np.float64)

    translation_matrix = np.array([[1.0, 0.0, -translate_x],
                                   [0.0, 1.0, -translate_y],
                                   [0.0, 0.0, 1.0]], dtype=np.float64)

    scale_matrix = np.array([[1. / max(scale_x, 1e-4), 0.0, 0.0],
                             [0.0, 1. / max(scale_y, 1e-4), 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)

    shear_matrix = np.array([[1.0, -shear_x, 0.0],
                             [-shear_y, 1.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)

    xmirror_matrix = np.array([[1.0 - 2 * mirror_x, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)

    ymirror_matrix = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0 - 2 * mirror_y, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)


    m = m @ rotation_matrix @ translation_matrix @ scale_matrix @ shear_matrix @ xmirror_matrix @ ymirror_matrix
    return m


def anchor_latent_space(G) -> None:

    if hasattr(G.synthesis, 'input'):
        shift = G.synthesis.input.affine(G.mapping.w_avg.unsqueeze(0))
        G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
        G.synthesis.input.affine.weight.data.zero_()


def force_fp32(G) -> None:

    G.synthesis.num_fp16_res = 0
    for name, layer in G.synthesis.named_modules():
        if hasattr(layer, 'conv_clamp'):
            layer.conv_clamp = None
            layer.use_fp16 = False




resume_specs = {

        'stylegan2': {

            'ffhq256':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-256x256.pkl',
            'ffhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhqu-1024x1024.pkl',
            'celebahq256':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl',
            'lsundog256':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-lsundog-256x256.pkl',
            'afhqcat512':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqcat-512x512.pkl',
            'afhqdog512':    'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqdog-512x512.pkl',
            'afhqwild512':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqwild-512x512.pkl',
            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-afhqv2-512x512.pkl',
            'brecahad512':   'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-brecahad-512x512.pkl',
            'cifar10':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-cifar10-32x32.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-metfacesu-1024x1024.pkl',

            'minecraft1024': 'https://github.com/jeffheaton/pretrained-gan-minecraft/releases/download/v1/minecraft-gan-2020-12-22.pkl',

            'afhqcat256':    'https://drive.google.com/u/0/uc?export=download&confirm=zFoN&id=1P9ouHIK-W8JTb6bvecfBe4c_3w6gmMJK',
            'anime256':      'https://drive.google.com/u/0/uc?export=download&confirm=6Uie&id=1EWOdieqELYmd2xRxUR4gnx7G10YI5dyP',
            'cub256':        'https://drive.google.com/u/0/uc?export=download&confirm=KwZS&id=1J0qactT55ofAvzddDE_xnJEY8s3vbo1_',
        },

        'stylegan3-r': {

            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-metfacesu-1024x1024.pkl',
        },

        'stylegan3-t': {

            'afhq512':       'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl',
            'ffhq1024':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl',
            'ffhqu1024':     'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl',
            'ffhqu256':      'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl',
            'metfaces1024':  'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfaces-1024x1024.pkl',
            'metfacesu1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl',

            'landscapes256': 'https://drive.google.com/u/0/uc?export=download&confirm=eJHe&id=14UGDDOusZ9TMb-pOrF0PAjMGVWLSAii1',  
            'wikiart1024':   'https://drive.google.com/u/0/uc?export=download&confirm=2tz5&id=18MOpwTMJsl_Z17q-wQVnaRLCUFZYSNkj',  
        }
}




def z_to_img(G, latents: torch.Tensor, label: torch.Tensor, truncation_psi: float, noise_mode: str = 'const') -> np.ndarray:

    assert isinstance(latents, torch.Tensor), f'latents should be a torch.Tensor!: "{type(latents)}"'
    if len(latents.shape) == 1:
        latents = latents.unsqueeze(0)
    img = G(z=latents, c=label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img + 1) * 255 / 2
    img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return img


def w_to_img(G, dlatents: Union[List[torch.Tensor], torch.Tensor], noise_mode: str = 'const', to_np: bool = True) -> np.ndarray:

    assert isinstance(dlatents, torch.Tensor), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
    if len(dlatents.shape) == 2:
        dlatents = dlatents.unsqueeze(0)

    synth_image = G.synthesis(dlatents, noise_mode=noise_mode)
    synth_image = (synth_image + 1) * 255/2
    if to_np:
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return synth_image


def get_w_from_seed(G, batch_sz, device, truncation_psi=1.0, seed=None, centroids_path=None, class_idx=None):


    if G.c_dim != 0:

        if class_idx is None:
            class_indices = np.random.RandomState(seed).randint(low=0, high=G.c_dim, size=(batch_sz))
            class_indices = torch.from_numpy(class_indices).to(device).to(torch.long)
            w_avg = G.mapping.w_avg.index_select(0, class_indices)
        else:
            w_avg = G.mapping.w_avg[class_idx].unsqueeze(0).repeat(batch_sz, 1).to(torch.long)
            class_indices = torch.full((batch_sz,), class_idx).to(device)

        labels = F.one_hot(class_indices, G.c_dim)

    else:
        w_avg = G.mapping.w_avg.unsqueeze(0)
        labels = None
        if class_idx is not None:
            print('Warning: --class is ignored when running an unconditional network')

    z = np.random.RandomState(seed).randn(batch_sz, G.z_dim)
    z = torch.from_numpy(z).to(device)
    w = G.mapping(z, labels)


    if centroids_path is not None:

        with dnnlib.util.open_url(centroids_path, verbose=False) as f:
            w_centroids = np.load(f)
        w_centroids = torch.from_numpy(w_centroids).to(device)
        w_centroids = w_centroids[None].repeat(batch_sz, 1, 1)


        dist = torch.norm(w_centroids - w[:, :1], dim=2, p=2)
        w_avg = w_centroids[0].index_select(0, dist.argmin(1))

    w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
    w = w_avg + (w - w_avg) * truncation_psi

    return w


def get_w_from_file(file, device='cuda', return_ext=False):
    filename, file_extension = os.path.splitext(file)
    assert file_extension in ['.npy', '.npz'], f'"{file}" has wrong file format! Use either ".npy" or ".npz"'

    if file_extension == '.npy':
        r = (np.load(file), '.npy') if return_ext else np.load(file)
    else:
        r = (np.load(file)['w'], '.npz') if return_ext else np.load(file)['w']
    return torch.from_numpy(r).to(device)





def save_config(ctx: click.Context, run_dir: Union[str, os.PathLike], save_name: str = 'config.json') -> None:

    with open(os.path.join(run_dir, save_name), 'w') as f:
        json.dump(ctx.obj, f, indent=4, sort_keys=True)





def make_run_dir(outdir: Union[str, os.PathLike], desc: str, dry_run: bool = False) -> str:


    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)


    if not dry_run:
        print('Creating output directory...')
        os.makedirs(run_dir)
    return run_dir
