
import torch.utils.data as data
import numpy as np

from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generator_tds import ChunkedGenerator

dad_metadata = {
    'layout_name': 'dad',
    'num_joints': 17,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ]
}


dad_wholebody_metadata = {
    'layout_name': 'dad',
    'num_joints': 133,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19]+list(range(33, 40))+list(range(46, 71))+list(range(91, 112)),
        [2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22]+list(range(24, 32))+list(range(40, 65))+list(range(112, 133)),
    ]
}

dad_metadata = {
    'layout_name': 'dad_wholebody',
    'num_joints': 85,
    'keypoints_symmetry': [
        [1, 3, 5, 7, 9, 11, 13, 15, 32-6, 33-6, 34-6, 35-6, 36-6, 37-6, 38-6, 39-6, 45-6, 46-6, 47-6, 48-6, 49-6, 65-6, 66-6, 67-6, 68-6, 69-6, 70-6], 
        [2, 4, 6, 8, 10, 12, 14, 16, 23-6, 24-6, 25-6, 26-6, 27-6, 28-6, 29-6, 30-6, 40-6, 41-6, 42-6, 43-6, 44-6, 59-6, 60-6, 61-6, 62-6, 63-6, 64-6]
    ]
}



class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True, MAE=False, tds=1):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        self.in_channels = opt.in_channels
        self.seq_start = opt.seq_start
        self.seq_length = opt.seq_length
        self.num_joints = dad_metadata["num_joints"]
        self.MAE=MAE
        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
                                                                                   subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all, MAE=MAE, tds=tds)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
                                                                                subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize // opt.stride, self.cameras_test, self.poses_test,
                                              self.poses_test_2d,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, MAE=MAE, tds=tds)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list):
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                anim = dataset[subject][action]
                
                if 'positions' in anim:
                    positions_3d = []
                    for cam in anim['cameras']:
                        # pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                        # Replace the previous line by:
                        pos_3d = anim['positions'][0]
                        pos_3d[:, 1:] -= pos_3d[:, :1] # Remove global offset, but keep trajectory in first position
                        positions_3d.append(pos_3d)
                    anim['positions_3d'] = positions_3d

    
        self.keypoints_name = 'gt_train'
        keypoints = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz',allow_pickle=True)
        keypoints_metadata = dad_metadata
        keypoints_symmetry = dad_metadata['keypoints_symmetry']

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        keypoints = keypoints['positions_2d'].item()

        for subject in folder_list:
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in dataset[subject].keys():
                assert action in keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                         subject)
                for cam_idx in range(len(keypoints[subject][action])):

                    mocap_length = dataset[subject][action]['positions_3d'][0].shape[0]
                    assert keypoints[subject][action].shape[0] >= mocap_length

                    if keypoints[subject][action].shape[0] > mocap_length:
                        keypoints[subject][action] = keypoints[subject][action][:mocap_length]

        for subject in keypoints.keys():
            for i, action in enumerate(keypoints[subject]):
               # for cam_idx, kps in enumerate(keypoints[subject][action]):
                kps = keypoints[subject][action].reshape((len(keypoints[subject][action]), dad_metadata['num_joints'], 3))
                j = i
                j = 0
                cam = dataset.cameras()[subject][j]
                if self.crop_uv == 0:
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                if self.in_channels == 2:
                    keypoints[subject][action] = kps[..., :2]
                else:
                    keypoints[subject][action] = kps
        
        return keypoints

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = np.array([self.keypoints[subject][action]])

                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]

                if subject in dataset.cameras():
                    cams = [dataset.cameras()[subject][0]] # One unique cam
                    # print("cams and pos 2d", len(cams), cams, len(poses_2d) )
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): 
                        pose_3d = poses_3d[i] #[self.seq_start: self.seq_start + self.seq_length]
                        pose_3d = np.reshape(pose_3d, (len(pose_3d), self.num_joints, 3))

                        out_poses_3d[(subject, action, i)] = pose_3d

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        return len(self.generator.pairs)
        #return 200

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        if self.MAE:
            cam, input_2D, action, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip,
                                                                                      reverse)
            if self.train == False and self.test_aug:
                _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
        else:
            cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)
        
            if self.train == False and self.test_aug:
                _, _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)

        if self.MAE:
            return cam, input_2D_update, action, subject, scale, bb_box, cam_ind
        else:
            return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, cam_ind



