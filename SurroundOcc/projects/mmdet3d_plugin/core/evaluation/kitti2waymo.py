# Copyright (c) OpenMMLab. All rights reserved.


try:
    from waymo_open_dataset import dataset_pb2 as open_dataset
    import mmcv
    import numpy as np
    import tensorflow as tf
    from glob import glob
    from os.path import join
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
except ImportError:

    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')




class KITTI2Waymo(object):


    def __init__(self,
                 kitti_result_files,
                 waymo_tfrecords_dir,
                 waymo_results_save_dir,
                 waymo_results_final_path,
                 prefix,
                 workers=64):

        self.kitti_result_files = kitti_result_files
        self.waymo_tfrecords_dir = waymo_tfrecords_dir
        self.waymo_results_save_dir = waymo_results_save_dir
        self.waymo_results_final_path = waymo_results_final_path
        self.prefix = prefix
        self.workers = int(workers)
        self.name2idx = {}
        for idx, result in enumerate(kitti_result_files):
            if len(result['sample_idx']) > 0:
                self.name2idx[str(result['sample_idx'][0])] = idx


        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.k2w_cls_map = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }

        self.T_ref_to_front_cam = np.array([[0.0, 0.0, 1.0, 0.0],
                                            [-1.0, 0.0, 0.0, 0.0],
                                            [0.0, -1.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]])

        self.get_file_names()
        self.create_folder()

    def get_file_names(self):

        self.waymo_tfrecord_pathnames = sorted(
            glob(join(self.waymo_tfrecords_dir, '*.tfrecord')))
        print(len(self.waymo_tfrecord_pathnames), 'tfrecords found.')

    def create_folder(self):

        mmcv.mkdir_or_exist(self.waymo_results_save_dir)

    def parse_objects(self, kitti_result, T_k2w, context_name,
                      frame_timestamp_micros):


        def parse_one_object(instance_idx):

            cls = kitti_result['name'][instance_idx]
            length = round(kitti_result['dimensions'][instance_idx, 0], 4)
            height = round(kitti_result['dimensions'][instance_idx, 1], 4)
            width = round(kitti_result['dimensions'][instance_idx, 2], 4)
            x = round(kitti_result['location'][instance_idx, 0], 4)
            y = round(kitti_result['location'][instance_idx, 1], 4)
            z = round(kitti_result['location'][instance_idx, 2], 4)
            rotation_y = round(kitti_result['rotation_y'][instance_idx], 4)
            score = round(kitti_result['score'][instance_idx], 4)



            y -= height / 2

            x, y, z = self.transform(T_k2w, x, y, z)


            heading = -(rotation_y + np.pi / 2)
            while heading < -np.pi:
                heading += 2 * np.pi
            while heading > np.pi:
                heading -= 2 * np.pi

            box = label_pb2.Label.Box()
            box.center_x = x
            box.center_y = y
            box.center_z = z
            box.length = length
            box.width = width
            box.height = height
            box.heading = heading

            o = metrics_pb2.Object()
            o.object.box.CopyFrom(box)
            o.object.type = self.k2w_cls_map[cls]
            o.score = score

            o.context_name = context_name
            o.frame_timestamp_micros = frame_timestamp_micros

            return o

        objects = metrics_pb2.Objects()

        for instance_idx in range(len(kitti_result['name'])):
            o = parse_one_object(instance_idx)
            objects.objects.append(o)

        return objects

    def convert_one(self, file_idx):

        file_pathname = self.waymo_tfrecord_pathnames[file_idx]
        file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')

        for frame_num, frame_data in enumerate(file_data):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(frame_data.numpy()))
            filename = f'{self.prefix}{file_idx:03d}{frame_num:03d}'

            for camera in frame.context.camera_calibrations:

                if camera.name == 1:
                    T_front_cam_to_vehicle = np.array(
                        camera.extrinsic.transform).reshape(4, 4)

            T_k2w = T_front_cam_to_vehicle @ self.T_ref_to_front_cam

            context_name = frame.context.name
            frame_timestamp_micros = frame.timestamp_micros

            if filename in self.name2idx:
                kitti_result = \
                    self.kitti_result_files[self.name2idx[filename]]
                objects = self.parse_objects(kitti_result, T_k2w, context_name,
                                             frame_timestamp_micros)
            else:
                print(filename, 'not found.(bevformer)')
                objects = metrics_pb2.Objects()

            with open(
                    join(self.waymo_results_save_dir, f'{filename}.bin'),
                    'wb') as f:
                f.write(objects.SerializeToString())

    def convert(self):

        print('Start converting ...')
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),
                                     self.workers)
        print('\nFinished ...')


        pathnames = sorted(glob(join(self.waymo_results_save_dir, '*.bin')))
        combined = self.combine(pathnames)

        with open(self.waymo_results_final_path, 'wb') as f:
            f.write(combined.SerializeToString())

    def __len__(self):

        return len(self.waymo_tfrecord_pathnames)

    def transform(self, T, x, y, z):

        pt_bef = np.array([x, y, z, 1.0]).reshape(4, 1)
        pt_aft = np.matmul(T, pt_bef)
        return pt_aft[:3].flatten().tolist()

    def combine(self, pathnames):

        combined = metrics_pb2.Objects()

        for pathname in pathnames:
            objects = metrics_pb2.Objects()
            with open(pathname, 'rb') as f:
                objects.ParseFromString(f.read())
            for o in objects.objects:
                combined.objects.append(o)

        return combined