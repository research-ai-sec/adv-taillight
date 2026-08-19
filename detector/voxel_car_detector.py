from scipy.ndimage import label, generate_binary_structure
import numpy as np
import torch


class CarInfo:

    def __init__(self, dict_init):
        self.car_idx = int(dict_init["car_idx"])
        self.num_features = int(dict_init["total"])
        self.min_x = int(dict_init["min_x"])
        self.max_x = int(dict_init["max_x"])
        self.min_y = int(dict_init["min_y"])
        self.max_y = int(dict_init["max_y"])
        self.min_z = int(dict_init["min_z"])
        self.max_z = int(dict_init["max_z"])
        if 'img_idx' in dict_init:
            self.img_idx = dict_init['img_idx']
        else:
            self.img_idx = None
        if 'scene_idx' in dict_init:
            self.scene_idx = dict_init['scene_idx']
        else:
            self.scene_idx = None
        if 'offset' in dict_init:
            self.offset = dict_init['offset']
        else:
            self.offset = None
        self.dic = dict_init

    def width(self):
        return self.max_x - self.min_x + 1

    def long(self):
        return self.max_y - self.min_y + 1

    def set_min_y(self, min_y):
        self.min_y = int(min_y)
        self.dic["min_y"] = int(min_y)

    def set_min_x(self, min_x):
        self.min_x = int(min_x)
        self.dic['min_x'] = int(min_x)

    def set_max_x(self, max_x):
        self.max_x = max_x
        self.dic['max_x'] = int(max_x)


    def get_2dmask(self):
        tmp = torch.zeros((200, 200), dtype=torch.int32)
        tmp[self.min_x:self.max_x+1, self.min_y:self.max_y+1] = 1
        return tmp


    def get_3dlabel(self):
        tmp = torch.zeros((200, 200, 16), dtype=torch.int32)
        tmp[self.min_x:self.max_x+1, self.min_y:self.max_y+1, self.min_z:self.max_z+1] = int(self.car_idx) + 1
        return tmp

    def get_3dmask(self):
        tmp = torch.zeros((200, 200, 16), dtype=torch.int32)

        tmp[self.min_x:self.max_x+1, self.min_y:self.max_y+1, self.min_z:self.max_z+1] = 1
        return tmp

    def __repr__(self):
        if self.img_idx is not None:
            return f'(id:{self.img_idx} x:{self.min_x},{self.max_x} y:{self.min_y},{self.max_y} z:{self.min_z}, {self.max_z})'
        return f'(x:{self.min_x},{self.max_x} y:{self.min_y},{self.max_y} z:{self.min_z}, {self.max_z})'

def detect_all_cars_with_ret(ret_voxel):
    if ret_voxel.shape != (200, 200, 16):
        raise ValueError("input tensor must be a 200x200x16 3D array")

    ret_voxel = ret_voxel.detach().cpu()
    car_mask = (ret_voxel == 4)
    structure = generate_binary_structure(3, 1)
    labeled_array, num_features = label(car_mask, structure=structure)


    cars_info_list = []
    for car_id in range(1, num_features + 1):

        coordinates = np.where(labeled_array == car_id)


        min_x, max_x = np.min(coordinates[0]), np.max(coordinates[0])
        min_y, max_y = np.min(coordinates[1]), np.max(coordinates[1])
        min_z, max_z = np.min(coordinates[2]), np.max(coordinates[2])


        car_idx = car_id - 1
        rank = {
            "car_idx": car_idx,
            "total": num_features,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z
        }
        cars_info_list.append(rank)
    return cars_info_list


def get_roi_mask(width=2.0):
    if width == 2.0:
        min_x, max_x = 98, 101
    elif width == 2.5:
        min_x, max_x = 97, 101
    elif width == 3.0:
        min_x, max_x = 97, 102
    min_y, max_y = 104, 113
    min_z, max_z = 6,9
    valid_range_mask = torch.zeros((200,200,16), dtype=torch.int32)
    valid_range_mask[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = 1

    return valid_range_mask


def downsample_3d_any(tensor, target_size=(100, 100, 8)):

    w, h, z = tensor.shape  
    target_w, target_h, target_z = target_size


    window_w = w // target_w
    window_h = h // target_h
    window_z = z // target_z

    result = torch.zeros(target_w, target_h, target_z, dtype=torch.bool)


    for i in range(target_w):
        w_start, w_end = i * window_w, (i + 1) * window_w
        for j in range(target_h):
            h_start, h_end = j * window_h, (j + 1) * window_h
            for k in range(target_z):
                z_start, z_end = k * window_z, (k + 1) * window_z


                window = tensor[w_start:w_end, h_start:h_end, z_start:z_end]


                if window.any():
                    result[i, j, k] = True

    return result.float()


def detect_valid_car_in_range(info_list, divide_car=True, return_others=False):
    valid_range_mask = get_roi_mask()
    info_3dlabel = torch.zeros((200,200,16), dtype=torch.int32)
    info_3dmask = torch.zeros((200,200,16), dtype=torch.int32)
    car_info_list = []
    for info in info_list:
        car_info = CarInfo(info)
        car_info_list.append(car_info)
        if car_info.min_y < 104:
            continue
        info_3dlabel += car_info.get_3dlabel()
        info_3dmask += car_info.get_3dmask()
    valid_info_label = info_3dlabel * valid_range_mask
    valid_val, counts2 = torch.unique(valid_info_label.flatten(), return_counts=True)
    index_list = [v-1  for v in valid_val if v > 0]

    selected_car = None
    if len(index_list) == 0:
        selected_car = None
    elif len(index_list) == 1:
        idx = index_list[0]
        selected_car = car_info_list[idx]
    else:
        selected_list = [car_info_list[i] for i in index_list if i < len(car_info_list)]
        if len(selected_list):
            selected_car = selected_list[0]
        else:
            sorted_by_min_y = sorted(selected_list, key=lambda carinfo: carinfo.min_y)
            back_car = sorted_by_min_y[0]
            front_car = sorted_by_min_y[1]
            if back_car.max_y <= front_car.min_y:
                selected_car = back_car
            else:
                sorted_by_center_x = sorted(selected_list, key=lambda carinfo: abs((carinfo.min_x + carinfo.max_x) / 2 - 100))
                selected_car = sorted_by_center_x[0]

    if divide_car and (selected_car is not None):
        car_width = selected_car.max_x - selected_car.min_x + 1
        if car_width >= 5.0 * 2.0:
            center_x = (selected_car.max_x + selected_car.min_x) / 2
            if center_x  < 100:
                new_min_x = selected_car.min_x + (car_width - (3 * 2))
                selected_car.set_min_x(new_min_x)
            else:
                new_max_x = selected_car.max_x - (car_width - (3 * 2))
                selected_car.set_max_x(new_max_x)


    if return_others:
        others = []
        for i, info in enumerate(info_list):
            if not (i in index_list):
                car_info = CarInfo(info)
                others.append(car_info)
        return selected_car, others

    return selected_car


def get_car_info(rank, voxel_size=0.5, occ_size=(200, 200, 16), lidar2front=2.5, car_id=0):
    min_x, max_x, min_y, max_y, min_z, max_z = rank
    center_x = occ_size[0] / 2
    center_y = occ_size[1] / 2 + lidar2front / voxel_size
    left = (min_x - center_x) * voxel_size
    right = (max_x - center_x) * voxel_size
    front = (max_y - center_y) * voxel_size
    back = (min_y - center_y) * voxel_size
    lateral_center = ((min_x + max_x) / 2 - center_x) * voxel_size


    size_x = max_x - min_x + 1
    size_y = max_y - min_y + 1
    size_z = max_z - min_z + 1


    height = size_z * voxel_size
    width = size_x * voxel_size
    length = size_y * voxel_size


    car_info = {
        "id": car_id,
        "voxel_range": {
            "x_min": min_x,
            "x_max": max_x,
            "y_min": min_y,
            "y_max": max_y,
            "z_min": min_z,
            "z_max": max_z,
            "size": (size_x, size_y, size_z)
        },
        "meter_range": {
            'left': left,
            'right': right,
            'front': front,
            'back': back,
            'height': height,
            'width': width,
            'length': length,
            'lateral_center': lateral_center
        }
    }

    return car_info


def find_rank_2d_mask(mask):
    coor = torch.where(mask == 1.0)
    min_x, max_x = coor[0].min().item(), coor[0].max().item()
    min_y, max_y = coor[1].min().item(), coor[1].max().item()
    return min_x, max_x, min_y, max_y

def split_with_mid_x(mask, mid_x, z_min, z_max):
    bev_mask = torch.zeros((200, 200), dtype=torch.float32)
    for layer in  range(mask.shape[-1]):
        layer_mask = mask[:, :, layer]
        bev_mask += layer_mask
    bev_mask[bev_mask > 0] = 1.0
    left_rect_bev_mask = torch.zeros_like(bev_mask)
    right_rect_bev_mask = torch.zeros_like(bev_mask)
    left_rect_bev_mask[0:mid_x + 1, :] = 1
    right_rect_bev_mask[mid_x + 1:, :] = 1
    left_rect_bev_mask = bev_mask * left_rect_bev_mask
    right_rect_bev_mask = bev_mask * right_rect_bev_mask
    diff = abs(torch.count_nonzero(left_rect_bev_mask) - torch.count_nonzero(right_rect_bev_mask))
    lmin_x, lmax_x, lmin_y, lmax_y = find_rank_2d_mask(left_rect_bev_mask)
    rmin_x, rmax_x, rmin_y, rmax_y = find_rank_2d_mask(right_rect_bev_mask)
    left_rank = lmin_x, lmax_x, lmin_y, lmax_y, z_min, z_max
    right_rank = rmin_x, rmax_x, rmin_y, rmax_y, z_min, z_max
    return left_rank, right_rank, diff

def split_to_2_rect(mask, x_min, x_max, z_min, z_max, lateral_center):
    bev_mask = torch.zeros((200, 200), dtype=torch.float32)
    for layer in  range(mask.shape[-1]):
        layer_mask = mask[:, :, layer]
        bev_mask += layer_mask
    bev_mask[bev_mask > 0] = 1.0

    def find_y_rank_in_x(bev_mask, x_value):

        row_data = bev_mask[x_value, :].flatten()

        y_indices = torch.where(row_data == 1.0)[0]

        if y_indices.numel() == 0:
            return None, None
        y_min = y_indices.min().item()
        y_max = y_indices.max().item()
        return y_min, y_max

    split_list = []
    for i in range(x_min, x_max + 1):
        y_min_i, y_max_i = find_y_rank_in_x(bev_mask, i)
        y_min_j, y_max_j = find_y_rank_in_x(bev_mask, i + 1)

        if y_min_i is None or y_min_j is None:
            continue
        if abs(y_min_i - y_min_j) > 1 or abs(y_max_i - y_max_j) > 1:
            left_rank, right_rank, diff  = split_with_mid_x(mask, i, z_min, z_max)
            info = i, left_rank, right_rank, diff
            split_list.append(info)


    valid_list = []
    diff_list = []
    i = 0
    for split_info in split_list:
        mid_x, left_rank, right_rank, diff = split_info
        left_car = get_car_info(left_rank)
        right_car = get_car_info(left_rank)

        if lateral_center < 0:
            if right_car["meter_range"]["back"] >= 0:
                valid_list.append(right_car)
                diff_list.append(diff)
        else:
            if left_car["meter_range"]["back"] >= 0:
                valid_list.append(left_car)
                diff_list.append(diff)
        i += 1

    if len(valid_list) == 1:

        return valid_list[0]
    elif len(valid_list) > 1:  

        best_diff = -1
        best_car = None
        for i in range(len(valid_list)):
            diff = diff_list[i]
            if best_car is None or diff < best_diff:
                best_car = valid_list[i]
                best_diff = diff
        return best_car
    else:

        for vaid_width in [3.0, 2.5, 2.0]:
            if lateral_center < 0:
                x_max_new = x_max
                x_min_new = max(x_min, int(x_max - vaid_width / 0.5 + 1))
            else:
                x_min_new = x_min
                x_max_new = min(x_max, int(x_min + vaid_width / 0.5 - 1))
            rect_bev_mask = torch.zeros_like(bev_mask)
            rect_bev_mask[x_min_new:x_max_new + 1, :] = 1
            rect_bev_mask = bev_mask * rect_bev_mask
            min_x, max_x, min_y, max_y = find_rank_2d_mask(rect_bev_mask)
            cur_rank = min_x, max_x, min_y, max_y, z_min, z_max
            cur_car = get_car_info(cur_rank)
            if cur_car["meter_range"]["back"] >= 0:
                break
        return cur_car


def detect_cars_in_3d_space(car_mask, occ_size=(200, 200, 16), voxel_size=0.5, lidar2front=2.5):


    if car_mask.shape != (200, 200, 16):
        raise ValueError("input tensor must be a 200x200x16 3D array")

    car_mask = car_mask.detach().cpu()


    car_mask = (car_mask == 4)


    structure = generate_binary_structure(3, 1)


    labeled_array, num_features = label(car_mask, structure=structure)


    cars_info_list = []


    for car_id in range(1, num_features + 1):

        coordinates = np.where(labeled_array == car_id)


        min_x, max_x = np.min(coordinates[0]), np.max(coordinates[0])
        min_y, max_y = np.min(coordinates[1]), np.max(coordinates[1])
        min_z, max_z = np.min(coordinates[2]), np.max(coordinates[2])


        rank = min_x, max_x, min_y, max_y, min_z, max_z
        car_info = get_car_info(rank, voxel_size, occ_size, lidar2front)
        car_info['id'] = car_id
        car_info['mask'] = labeled_array == car_id

        cars_info_list.append(car_info)

    if len(cars_info_list) == 0:
        print("car not found")


    closest_car = None
    min_back = None
    for car in cars_info_list:
        back = car['meter_range']['back']
        front = car['meter_range']['front']
        lateral_center = car['meter_range']['lateral_center']
        left = car['meter_range']['left']
        right = car['meter_range']['right']
        mask = car['mask']
        if (back >= -0.5) and (left <= 0.5 and right >= -0.5):  
            if min_back  is None or abs(back) < abs(min_back):
                min_back = back

                car_min_x, car_max_x = car['voxel_range']['x_min'], car['voxel_range']['x_max']
                if car_max_x - car_min_x + 1 >= 5.0 / voxel_size:


                    if lateral_center >= 0:
                        min_x = car['voxel_range']['x_min']
                        max_x = min_x + 3.0 / voxel_size - 1
                        mid_x = int(max_x)
                        left_rank, right_rank, _ = split_with_mid_x(mask, mid_x, car['voxel_range']['z_min'], car['voxel_range']['z_max'])
                        best_rank = left_rank
                    else: 
                        max_x = car['voxel_range']['x_max']
                        min_x = max_x - 3.0 / voxel_size + 1
                        mid_x = int(min_x - 1)
                        left_rank, right_rank, _ = split_with_mid_x(mask, mid_x, car['voxel_range']['z_min'], car['voxel_range']['z_max'])
                        best_rank = right_rank
                    closest_car = get_car_info(best_rank, voxel_size, occ_size, lidar2front, car_id=car['id'])
                else:
                    closest_car = car
        elif (back <=0 and front >= 0) and (left <= 0 and right >= 0):
            closest_car = split_to_2_rect(mask, car['voxel_range']['x_min'], car['voxel_range']['x_max'], car['voxel_range']['z_min'], car['voxel_range']['z_max'], lateral_center)
            close_back = closest_car['meter_range']['back']
            if close_back < 0:
                print_car_info('closed',closest_car)


    if closest_car is None:
        for cinf in cars_info_list:
            back = cinf['meter_range']['back']
            if back >= -1.0 and (left <= 1.0 and right >= -1.0):
                closest_car = cinf
        if closest_car is None:
            print(f'closest_car not found')
    return (num_features, cars_info_list, closest_car)


def print_car_info(message,car_info):
    ret_dir = {}
    for key, value in car_info.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                ret_dir[f"{sub_key}"] = sub_value
        else:
                ret_dir[key] = value
    keys = ['file_name','id', 'h', 'd', 'r','x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max',
            'size_x', 'size_y', 'size_z', 'left', 'right', 'front', 'back', 'height', 'width', 'length', 'lateral_center']
    write_info = {k: ret_dir[k] for k in keys if k in ret_dir}
    print(f"{message}: {write_info}\n")


def get_sumoflabel_with_car_list(info_dict_list):
    info_3dlabel = torch.zeros((200,200,16), dtype=torch.int32)
    info_3dmask = torch.zeros((200,200,16), dtype=torch.int32)
    car_info_list = []
    for info in info_dict_list:
        if not isinstance(info, CarInfo):
            car_info = CarInfo(info)
        else:
            car_info = info
        car_info_list.append(car_info)
        if car_info.min_y < 104 and car_info.max_y > 104:
            car_info.set_min_y(104)
        info_3dlabel += car_info.get_3dlabel()
        info_3dmask += car_info.get_3dmask()
    return car_info_list, info_3dmask, info_3dlabel
