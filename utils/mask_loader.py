import json
import cv2
import numpy as np

class MaskLoader:
    def __init__(self, mask_path, width, height):
        self.mask_path = mask_path
        self.width = width
        self.height = height

    def load(self):
        label_map = {}
        with open(self.mask_path, 'r') as f:
            data = json.load(f)
            shapes = data['shapes']
            for label in shapes:
                name = label['label']
                points = label['points']
                label_map[name] = points
        self.label_map = label_map
        self.data = data
        org_points = label_map['rect']
        self.points = [[int(round(x)) for x in sublist] for sublist in org_points]

        if 'light_bg_color' in data:
            self.bg_color = data['light_bg_color']['RGB']
            self.light_color = data['light_color']['RGB']
        else:
            self.bg_color = None
            self.light_color = None
        return self.points

    @staticmethod
    def get_mask_st(points, width, height):

        height, width= int(height), int(width)
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.fillPoly(mask, [points], [255,255,255])
        return mask

    def get_mask(self):
        points = self.load()
        return MaskLoader.get_mask_st(points, self.width, self.height)
