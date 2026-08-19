from PIL import Image, ImageOps
import torch
import numpy as np
import cv2
import math
import torchvision
import random
import os

def to_binary_mask(mask):
    mask = (mask / 255.0).astype(np.uint8)
    mask = np.stack([mask] * 3, axis=-1)
    return mask


def move_and_scale_to_bottom(mask, offset, scale):


    offset = max(0, int(offset))


    scale = max(0.01, min(1.0, float(scale)))


    h, w, c = mask.shape


    result = np.zeros_like(mask)


    move_down = np.zeros_like(mask)
    if offset > 0:

        move_down[offset:, :, :] = mask[:h-offset, :, :]
    else:
        move_down = mask.copy()


    new_h = max(1, int(h * scale))


    crop_top = max(0, h - new_h - offset)


    if crop_top > 0:
        crop_img = move_down[crop_top:, :, :]
    else:
        crop_img = move_down


    scaled_img = cv2.resize(
        crop_img.astype(np.float32),
        (w, new_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(mask.dtype)


    start_row = h - new_h


    if start_row >= 0 and start_row < h:

        place_height = min(new_h, h - start_row)
        result[start_row:start_row+place_height, :, :] = scaled_img[:place_height, :, :]

    return result

def move_and_scale_if_need(h, d, r, src_dir, car_img, car_mask, rect_mask, ring_img, ring_mask, screen_mask):
    if h != 1.5:
        ref_file_name = f"h_1.50_d_{d:.2f}_r_{r:.2f}"
        ref_car_mask = cv2.imread(os.path.join(src_dir, ref_file_name, "car_mask.png"), cv2.IMREAD_GRAYSCALE)
        ref_car_mask = to_binary_mask(ref_car_mask)

        ref_min_x, ref_max_x,ref_min_y, ref_max_y =  get_mask_bounds(ref_car_mask)
        x_min, x_max, y_min, y_max = get_mask_bounds(car_mask)
        ref_width = ref_max_x - ref_min_x + 1
        mask_width = x_max - x_min + 1
        offset_bottom = ref_max_y - y_max
        scale = ref_width / mask_width

        car_mask = move_and_scale_to_bottom(car_mask, offset_bottom, scale)
        car_img = move_and_scale_to_bottom(car_img, offset_bottom, scale)
        rect_mask = move_and_scale_to_bottom(rect_mask, offset_bottom, scale)
        screen_mask = move_and_scale_to_bottom(screen_mask, offset_bottom, scale)
        ring_img = move_and_scale_to_bottom(ring_img, offset_bottom, scale)
        ring_mask = move_and_scale_to_bottom(ring_mask, offset_bottom, scale)

    return car_img, car_mask, rect_mask, ring_img, ring_mask, screen_mask


def mask_to_tensor(img, device):
    img = (img).astype(np.float32)
    img = np.stack([img] * 3, axis=0)
    tensor = torch.from_numpy(img).to(device)
    return tensor

def img_to_tensor(img, device):
    img_ts = torch.from_numpy(img)
    img_ts = img_ts.permute(2, 0, 1)
    img_ts = img_ts.float().to(device)
    return img_ts

def load_all_from_vir_car_dir(h, d, r, src_dir, file_name, device, to_tensor=True):
    dir_str = os.path.join(src_dir, file_name)
    car_img = cv2.imread(os.path.join(dir_str, "car.png"), cv2.IMREAD_COLOR)
    ring_img = cv2.imread(os.path.join(dir_str, "ring.png"), cv2.IMREAD_COLOR)
    car_mask = cv2.imread(os.path.join(dir_str, "car_mask.png"), cv2.IMREAD_GRAYSCALE)
    screen_mask = cv2.imread(os.path.join(dir_str, "screen.png"), cv2.IMREAD_GRAYSCALE)
    rect_mask = cv2.imread(os.path.join(dir_str, "rect.png"), cv2.IMREAD_GRAYSCALE)
    ring_mask = cv2.imread(os.path.join(dir_str, "ring_mask.png"), cv2.IMREAD_GRAYSCALE)

    if car_img is None:
        raise ValueError(f"❌ Error: car_image is None when load path={dir_str}!")

    if rect_mask is None:
        raise ValueError(f"❌ Error: car_image is None when load path={dir_str}!")


    car_mask = to_binary_mask(car_mask)
    screen_mask = to_binary_mask(screen_mask)
    rect_mask = to_binary_mask(rect_mask)
    ring_mask = to_binary_mask(ring_mask)

    car_img, car_mask, rect_mask, ring_img, ring_mask, screen_mask = move_and_scale_if_need(h, d, r, src_dir, car_img, car_mask, rect_mask, ring_img, ring_mask, screen_mask)

    if car_img is None:
        raise ValueError(f"❌ Error: car_image is None after move_and_scale_if_need")

    if rect_mask is None:
        raise ValueError(f"❌ Error: rect_mask is None after move_and_scale_if_need")

    if to_tensor:
        car_mask = img_to_tensor(car_mask, device)
        screen_mask = img_to_tensor(screen_mask, device)
        rect_mask = img_to_tensor(rect_mask, device)
        ring_mask = img_to_tensor(ring_mask, device)
        car_img = img_to_tensor(car_img, device)
        ring_img = img_to_tensor(ring_img, device)

    if car_img is None:
        raise ValueError(f"❌ Error: car_image is None after img_to_tensor")


    if rect_mask is None:
        raise ValueError(f"❌ Error: rect_mask is None after mask_to_tensor")
    return [car_img, car_mask, screen_mask, rect_mask, ring_img, ring_mask]


def max_brightness_preserve_contrast(img):

    max_val = 255
    if isinstance(img, torch.Tensor):

        current_min = img.min()
        current_max = img.max()


        stretched = ((img - current_min) / (current_max - current_min)) * max_val
        stretched = torch.clamp(stretched, 0, 255).to(torch.uint8)
        return stretched.to(img.dtype)
    else:

        current_min = np.min(img)
        current_max = np.max(img)


        stretched = ((img - current_min) / (current_max - current_min)) * max_val
        stretched = np.clip(stretched, 0, 255).astype(np.uint8)
        return stretched.astype(img.dtype)


def resize_image(image, target_width, target_height):

    ret = ImageOps.fit(image, (target_width,target_height), Image.Resampling.LANCZOS)
    return ret


def get_mask_bounds(mask):

    if len(mask.shape) == 5:
        mask = mask[0][0]
    if len(mask.shape) == 4:
        mask = mask[0]

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = mask[:, :, 0]
    if len(mask.shape) == 3 and mask.shape[0] == 3:
        mask = mask[0]


    if isinstance(mask, torch.Tensor):
        coords = torch.argwhere(mask == 1)
    elif isinstance(mask, np.ndarray):
        coords = np.argwhere(mask == 1)

    if coords.size == 0:

        return None, None, None, None


    y_min = coords[:, 0].min()  
    y_max = coords[:, 0].max()  
    x_min = coords[:, 1].min()  
    x_max = coords[:, 1].max()

    return x_min, x_max, y_min, y_max


def generate_positions(
    num_images: int = 10,
    image_width: int = 100,
    total_width: int = 1200,
    max_offset_ratio: float = 0.5,
    max_attempts: int = 100
    ):


    total_gap = total_width - num_images * image_width
    if total_gap < 0:
        raise ValueError("total width is insufficient to hold all images")
    S = total_gap / (num_images - 1)
    best_positions = []
    best_offsets = []
    attempt_records = []
    min_total_overlap = float('inf')
    max_offset = max_offset_ratio * image_width
    for attempt in range(max_attempts):

        random_values = np.random.uniform(-max_offset, max_offset, num_images - 2)
        offsets = [0.0] + list(random_values) + [0.0]

        gap_changes = [offsets[i+1] - offsets[i] for i in range(num_images - 1)]
        new_gaps = [S + change for change in gap_changes]

        total_overlap = sum(abs(gap) for gap in new_gaps if gap < 0)

        positions = [0.0]
        for i in range(num_images - 1):
            positions.append(positions[-1] + image_width + new_gaps[i])

        record = {            
        "attempt": attempt + 1,            
        "positions": positions.copy(),            
        "offsets": offsets.copy(),            
        "total_overlap": total_overlap,            
        "gaps": new_gaps.copy()        
        }
        attempt_records.append(record)

        if total_overlap == 0:  
            best_positions = positions
            best_offsets = offsets
            break
        elif total_overlap < min_total_overlap:
            min_total_overlap = total_overlap            
            best_positions = positions
            best_offsets = offsets
    return best_positions

def composite_images(mask, imgs_tensor, scale=1.0, fix_width=0, p_rotation=0.5, rotation_max=10, n_gaps=0, max_offset=0.5, points=None, angles=None):

    def tensor2int(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.item()

        return int(round(tensor))


    x_min, x_max, y_min, y_max = get_mask_bounds(mask)
    x_min = tensor2int(x_min)
    x_max = tensor2int(x_max)
    y_min = tensor2int(y_min)
    y_max = tensor2int(y_max)
    b, c, h, w = imgs_tensor.shape


    img_target_size = y_max - y_min
    fix_width = int(fix_width * img_target_size / h)
    canvas_width  = x_max - x_min
    canvas_height = img_target_size
    canvas = torch.zeros((3, canvas_height, canvas_width ),
                        dtype=torch.float32, device=mask.device)


    if scale != 1.0:
        mid_height = math.ceil(img_target_size * scale)
        mid_width  = math.ceil(img_target_size * scale)
        imgs_tensor = torch.nn.functional.interpolate(imgs_tensor, size=(mid_height, mid_width), mode='area')
        imgs_tensor = torch.nn.functional.interpolate(imgs_tensor, size=(img_target_size, img_target_size), mode='bilinear')
    else:
        imgs_tensor = torch.nn.functional.interpolate(imgs_tensor, size=(img_target_size, img_target_size), mode='bilinear')

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
    if angles is not None and len(angles) == len(imgs_tensor):
        angle_list = angles
    else:
        angle_list = []
        for i in range(len(imgs_tensor)):
            if random.random() < p_rotation:
                angle = random.uniform(-rotation_max, rotation_max)
                angle_list.append(angle)
            else:
                angle_list.append(0)


    for idx, img in enumerate(imgs_tensor):
        start_width = int(fix_width)
        end_width = int(img.shape[2] - fix_width)
        cropped_tensor = img[:, :, start_width:end_width]
        r = cropped_tensor[0,:,:]
        g = cropped_tensor[1,:,:]
        b = torch.zeros_like(r)  
        bgr = torch.stack([b, g, r], dim=0)

        angle = angle_list[idx]
        if angle != 0:
            img_rotated = torchvision.transforms.functional.rotate(bgr, angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        else:
            img_rotated = bgr

        pos_x = int(x_points[idx])
        end_x = pos_x + img_target_size

        if end_x > canvas_width:
            img_rotated = img_rotated[:, :, :(canvas_width - pos_x)]
            end_x = canvas_width

        if pos_x < canvas_width and end_x > pos_x:
            canvas[:, :, pos_x:end_x] = img_rotated[:, :, :(end_x - pos_x)]


    padded_img = torch.zeros_like(mask, dtype=torch.float32)
    padded_img[0, :, :, y_min:y_min+canvas_height, x_min:x_min+canvas_width] = canvas

    return padded_img, x_points, angle_list, imgs_tensor


def calulate_scale_factor(mask, target_count=5000):

    if len(mask.shape) == 5:
        mask = mask[0][0][0]
    if len(mask.shape) == 4:
        mask = mask[0][0]
    elif len(mask.shape) == 3:
        mask = mask[0]
    count = torch.count_nonzero()
    scale_factor = torch.sqrt(count / target_count)
    return scale_factor