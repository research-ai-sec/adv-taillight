import os
import torch
import dnnlib
import legacy
from torch_utils.ops import conv2d_gradfix
import copy
from torch_utils import misc
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import cv2
from utils.mask_loader import MaskLoader
import math
from pg_modules.discriminator import ProjectedDiscriminator


def get_w_from_seed(G, z, batch_sz, device, truncation_psi=1.0, seed=None, centroids_path=None, class_idx=None):


    if G.c_dim != 0:

        if class_idx is None:
            class_indices = np.random.RandomState(seed).randint(low=0, high=G.c_dim, size=(batch_sz))
            class_indices = torch.from_numpy(class_indices).to(device).to(torch.long)
            w_avg = G.mapping.w_avg.index_select(0, class_indices)
        else:
            class_indices = class_idx.to(device).to(torch.long)
            w_avg = G.mapping.w_avg.index_select(0, class_indices)

        labels = F.one_hot(class_indices, G.c_dim)

    else:
        w_avg = G.mapping.w_avg.unsqueeze(0)
        labels = None
        if class_idx is not None:
            print('Warning: --class is ignored when running an unconditional network')

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


def load_generator_styleganxl(network_pkl):
    with dnnlib.open_url(network_pkl, verbose=True) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema']
        if 'D' in network_dict:
            D = network_dict['D']
        else:
            D = None

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    conv2d_gradfix.enabled = True

    G = copy.deepcopy(G).eval().requires_grad_(False)
    if D is not None:
        D = copy.deepcopy(D).eval().requires_grad_(False)
        backbone_kwargs = dnnlib.EasyDict()
        backbone_kwargs.cout = 64
        backbone_kwargs.expand = True
        backbone_kwargs.proj_type = 2
        backbone_kwargs.num_discs = 4
        backbone_kwargs.cond = True
        backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0']
        diffaug=True
        interp224= True
        projd = ProjectedDiscriminator(backbones,diffaug,interp224,backbone_kwargs)
        projd.eval().requires_grad_(False)
        with torch.no_grad():
            misc.copy_params_and_buffers(D, projd, require_all=False)
    else:
        projd = None

    return G, projd


def genere_img(network_pkl, batch_gen, device):
    G, D = load_generator_styleganxl(network_pkl)
    G = G.to(device)
    z = np.random.RandomState(None).randn(batch_gen, G.z_dim)
    z = torch.from_numpy(z).to(device)
    if G.c_dim != 0:
        class_idx = torch.randint(0, G.c_dim, (batch_gen,)).to(device)
    else:
        class_idx = None
    w = get_w_from_seed(G, z, batch_gen, device,class_idx=class_idx)
    imgs = G.synthesis(w)
    imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)


    return imgs

def get_mask_bounds(mask):

    if len(mask.shape) == 5:
        mask = mask[0][0]
    if len(mask.shape) == 4:
        mask = mask[0]

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = mask[:, :, 0]
    if len(mask.shape) == 3 and mask.shape[0] == 3:
        mask = mask[0]

    coords = torch.argwhere(mask == 1)

    if coords.size == 0:

        return None, None, None, None


    y_min = coords[:, 0].min()
    y_max = coords[:, 0].max()
    x_min = coords[:, 1].min()
    x_max = coords[:, 1].max()

    return x_min, x_max, y_min, y_max


def composite_images(mask, imgs_np, scale=1.0, fix_width=0):


    x_min, x_max, y_min, y_max = get_mask_bounds(mask)


    target_height = y_max - y_min
    scale_rate = target_height / imgs_np[0].shape[1]  
    single_width = (imgs_np[0].shape[2] - fix_width * 2) * scale_rate  
    target_width = single_width * len(imgs_np)  
    if isinstance(target_width, torch.Tensor):
        target_width = target_width.item()
    if isinstance(target_height, torch.Tensor):
        target_height = target_height.item()
    if isinstance(target_width, float):
        target_width = math.ceil(target_width)
    if isinstance(target_height, float):
        target_height = math.ceil(target_height)


    bgt_list = []
    for img in imgs_np:
        start_width = int(fix_width)
        end_width = int(img.shape[2] - fix_width)
        cropped_tensor = img[:, :, start_width:end_width]
        r = cropped_tensor[0,:,:]
        g = cropped_tensor[1,:,:]

        b = torch.zeros_like(r)
        bgr = torch.stack([b, g, r], dim=0)
        bgt_list.append(bgr)
    concatenated_img = torch.cat(bgt_list, dim=-1).unsqueeze(0).to(torch.float32)
    if scale != 1.0:
        mid_height = math.ceil(target_height * scale)
        mid_width  = math.ceil(target_width * scale)
        concatenated_img = torch.nn.functional.interpolate(concatenated_img, size=(mid_height, mid_width), mode='area')
        concatenated_img = torch.nn.functional.interpolate(concatenated_img, size=(target_height, target_width), mode='bilinear')
    else:
        concatenated_img = torch.nn.functional.interpolate(concatenated_img, size=(target_height, target_width), mode='bilinear')


    padded_img = torch.zeros_like(mask, dtype=torch.float32)


    start_x = x_min + torch.div(x_max - x_min - target_width, 2, rounding_mode='trunc')
    start_y = y_min
    end_y = start_y + target_height
    end_x = start_x + target_width


    start_x = start_x if start_x >= x_min else x_min
    end_x = end_x if end_x <= x_max else x_max
    end_y = end_y if end_y <= y_max else y_max


    padded_img[0, 0, :, start_y:end_y, start_x:end_x] = concatenated_img


    return padded_img


def grabcut_segmen_mask(image_list):
    mask_list = []
    for image in image_list:

        image = image.permute(1, 2, 0).detach().cpu().numpy()
        image = image.astype(np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (2, 2, image.shape[1] -4, image.shape[0]-4)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)


        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask_bg_np = np.expand_dims(mask, axis=0).repeat(3, axis=0)
        mask_bg_tensor = torch.from_numpy(mask_bg_np).to(image_list.dtype).to(image_list.device)
        mask_list.append(mask_bg_tensor)
    mask_ret = torch.stack(mask_list, dim=0)

    return  mask_ret

def do_styleganxl_forward():


    current_file_path = Path(__file__).resolve()
    root_path = current_file_path.parent.parent

    network_pkl = os.path.join(root_path,'data', 'weight', 'styleganxl_imagenet32.pkl')
    device =  torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    batch_gen = 1
    imgs_org = genere_img(network_pkl, batch_gen, device)
    masked_bg = grabcut_segmen_mask(imgs_org)
    imgs = imgs_org * masked_bg
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img.astype(np.uint8)
        save_dir = os.path.join(root_path, 'output', 'syn-images')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'remove.png'), img)

        img_org = imgs_org[i]
        img_org = img_org.permute(1, 2, 0).cpu().numpy()
        img_org = img_org.astype(np.uint8)
        save_dir = os.path.join(root_path, 'output', 'syn-images')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'org.png'), img_org)


    current_file_path = Path(__file__).resolve()
    root_path = current_file_path.parent.parent
    car_img_path = os.path.join(root_path,'data', 'images', 'l7-blue-seg.png')
    car_img = cv2.imread(car_img_path)
    mask_path = car_img_path.replace('png', 'json')
    width, height = car_img.shape[1], car_img.shape[0]
    mask_loader = MaskLoader(mask_path, width, height)
    mask = mask_loader.get_mask()
    mask_points = mask_loader.points
    mask = (mask / 255.0).astype(np.uint8)
    mask = torch.from_numpy(mask).to(torch.float32)
    mask = mask.permute(2, 0, 1)
    mask = mask.unsqueeze(0)


    batch_mask = torch.zeros((6, *mask.shape[1:]), dtype=mask.dtype, device=mask.device)
    batch_mask[0] = mask
    if len(mask.shape) == 4:
        batch_mask = batch_mask.unsqueeze(0)

    car_img_tensor = torch.from_numpy(car_img).permute(2, 0, 1).to(torch.float32)
    batch_car_img = torch.zeros((6, *car_img_tensor.shape), dtype=mask.dtype, device=mask.device)
    batch_car_img[0] = car_img_tensor
    batch_car_img= batch_car_img.unsqueeze(0)

    composite_img_tensor = composite_images(batch_mask, imgs)
    ret_img_tensor = batch_car_img * (1 - mask) + composite_img_tensor * mask
    ret_img = ret_img_tensor[0][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    composite_img = composite_img_tensor[0][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)


    cv2.imshow("Composite Image", composite_img)
    cv2.imshow("result Image", ret_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    do_styleganxl_forward()
