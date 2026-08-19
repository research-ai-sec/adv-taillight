import os, sys
import cv2, imageio
import mayavi.mlab as mlab
import numpy as np
import torch


colors = np.array(
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
        [0, 255, 127, 255],
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)



voxel_size = 0.5
pc_range = [-50, -50,  -5, 50, 50, 3]

visual_path = sys.argv[1]
fov_voxels = np.load(visual_path)

fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
fov_voxels[:, 0] += pc_range[0]
fov_voxels[:, 1] += pc_range[1]
fov_voxels[:, 2] += pc_range[2]



figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))

plt_plot_fov = mlab.points3d(
    fov_voxels[:, 0],
    fov_voxels[:, 1],
    fov_voxels[:, 2],
    fov_voxels[:, 3],
    colormap="viridis",
    scale_factor=voxel_size - 0.05*voxel_size,
    mode="cube",
    opacity=1.0,
    vmin=0,
    vmax=19,
)


plt_plot_fov.glyph.scale_mode = "scale_by_vector"
plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors



mlab.show()
