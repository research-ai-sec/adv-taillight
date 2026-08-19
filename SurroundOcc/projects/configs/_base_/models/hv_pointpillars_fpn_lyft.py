_base_ = './hv_pointpillars_fpn_nus.py'






model = dict(
    pts_voxel_layer=dict(
        max_num_points=20,
        point_cloud_range=[-80, -80, -5, 80, 80, 3],
        max_voxels=(60000, 60000)),
    pts_voxel_encoder=dict(
        feat_channels=[64], point_cloud_range=[-80, -80, -5, 80, 80, 3]),
    pts_middle_encoder=dict(output_shape=[640, 640]),
    pts_bbox_head=dict(
        num_classes=9,
        anchor_generator=dict(
            ranges=[[-80, -80, -1.8, 80, 80, -1.8]], custom_values=[]),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7)),

    train_cfg=dict(pts=dict(code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])))
