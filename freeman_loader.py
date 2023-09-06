# coding=utf-8
# Copyright 2020 The Google AI Perception Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FreeMan Dataset Loader."""
import json
import os
import pickle

import aniposelib
import numpy as np
import cv2


class FreeMan:
    """A dataset class for loading, processing and plotting FreeMan."""
    VIEWS = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08']

    def __init__(self, base_dir, fps=30, split=''):
        assert os.path.exists(base_dir), f'Data does not exist at {base_dir}!'

        # Init paths
        sub_dir = os.path.join(base_dir, f'{int(fps)}FPS')
        self.video_dir = os.path.join(sub_dir, 'videos/')
        self.camera_dir = os.path.join(sub_dir, 'cameras/')
        self.motion_dir = os.path.join(sub_dir, 'motions/')
        self.keypoints2d_dir = os.path.join(sub_dir, 'keypoints2d/')      # kpts2d_reproj_original
        self.keypoints3d_dir = os.path.join(sub_dir, 'keypoints3d/')
        self.bbox2d_dir = os.path.join(sub_dir, 'bbox2d/')
        self.filter_file = os.path.join(sub_dir, 'ignore_list.txt')
        if len(split) == 0:
            self.session_list = self._sessionfile_to_list(os.path.join(sub_dir, 'session_list.txt'))
        elif split == 'train':
            self.session_list = self._sessionfile_to_list(os.path.join(sub_dir, 'train.txt'))
        elif split == 'validation':
            self.session_list = self._sessionfile_to_list(os.path.join(sub_dir, 'validation.txt'))
        elif split == 'test':
            self.session_list = self._sessionfile_to_list(os.path.join(sub_dir, 'test.txt'))

    def _sessionfile_to_list(self, filepath):
        with open(filepath, 'r') as fr:
            lines = fr.readlines()
        return [item.strip() for item in lines]

    def get_children_sessions(self, session_shortcut, num=10):
        children_list = list()
        for session in self.session_list:
            if session_shortcut in session:
                children_list.append(session)
        children_list = sorted(children_list)
        return children_list[: min(num, len(children_list))]

    def get_parent_name(cls, video_path):
        video_name = os.path.basename(video_path).split('.')[0]
        camidx = int(video_name.split('c')[1][1:])
        session_name = video_name.split('_')[0]
        return session_name, camidx

    def load_cgroup(self, session_name):
        pass

    def get_video_path(self, session_name, cam):
        """get full path of video"""
        assert cam >= 1 and cam <=8, ValueError('Illegal Camera Index. Shall be 1 ~ 8')
        assert session_name in self.session_list, ValueError(f'{session_name} not found in sessions!')
        if os.path.isfile(os.path.join(self.video_dir, session_name, 'vframes', f'c0{cam}.mp4')):
            return os.path.join(self.video_dir, session_name, 'vframes', f'c0{cam}.mp4')
        else:
            return None

    @classmethod
    def load_camera_group(cls, camera_dir, session_name):
        """Load a set of cameras in the environment."""
        file_path = os.path.join(camera_dir, f'{session_name}.json')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        with open(file_path, 'r') as f:
            params = json.load(f)
        cameras = []
        for param_dict in params:
            camera = aniposelib.cameras.Camera(name=param_dict['name'],
                                               size=param_dict['size'],
                                               matrix=param_dict['matrix'],
                                               rvec=param_dict['rotation'],
                                               tvec=param_dict['translation'],
                                               dist=param_dict['distortions'])
            cameras.append(camera)
        camera_group = aniposelib.cameras.CameraGroup(cameras)
        return camera_group, params

    @classmethod
    def load_bbox2d(cls, bbox_dir, session_name):
        """Load a 2D keypoint sequence represented using COCO format."""
    
        bbox_path = os.path.join(bbox_dir, f'{session_name}.npy')
        assert os.path.exists(bbox_path), f'File {bbox_path} does not exist!'
        
        bboxes = np.load(bbox_path, allow_pickle=True)
        return bboxes

    @classmethod
    def load_keypoints2d(cls, keypoint_dir, session_name, bbox_dir=None, key='keypoints2d'):
        """Load a 2D keypoint sequence represented using COCO format."""
        file_path = os.path.join(keypoint_dir, f'{session_name}.npy')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        
        data = np.load(file_path, allow_pickle=True)
        keypoints2d = data[0][key]        # (nviews, N, 17, 3)
        kpts_center = data[0]['center']
        kpts_scale = data[0]['scale']

        if bbox_dir is not None:
            bbox_path = os.path.join(bbox_dir, f'{session_name}.npy')
            assert os.path.exists(bbox_path), f'File {bbox_path} does not exist!'
            
            bboxes = np.load(bbox_path, allow_pickle=True)
            return keypoints2d, kpts_center, kpts_scale, bboxes
        return keypoints2d, kpts_center, kpts_scale

    @classmethod
    def load_keypoints3d(cls, keypoint_dir, seq_name, use_optim=True, use_smooth=True):
        """Load a 3D keypoint sequence represented using COCO format."""
        file_path = os.path.join(keypoint_dir, f'{seq_name}.npy')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        
        data = np.load(file_path, allow_pickle=True)
        if use_optim:
            return data[0]['keypoints3d_optim']                # (N, 17, 3)
        elif use_smooth:
            try:
                return data[0]['keypoints3d_smoothnet32']      # (N, 17, 3)
            except KeyError:
                if 'keypoints3d_smoothnet' in data[0]:
                    return data[0]['keypoints3d_smoothnet']    # (N, 17, 3)
                else:
                    return data[0]['keypoints3d_optim']        # (N, 17, 3)
        else:
            return data[0]['keypoints3d']                      # (N, 17, 3)

    @classmethod
    def load_motion(cls, motion_dir, seq_name):
        """Load a motion sequence represented using SMPL format."""
        # file_path = os.path.join(motion_dir, f'{seq_name}.pkl')
        file_path = os.path.join(motion_dir, f'{seq_name}.npy')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        # with open(file_path, 'rb') as f:
        #     data = pickle.load(f)
        data = np.load(file_path, allow_pickle=True)[0]
        smpl_poses = data['smpl_poses']  # (N, 24, 3)
        smpl_scaling = data['smpl_scaling']  # (1,)
        smpl_trans = data['smpl_transl']  # (N, 3)
        return smpl_poses, smpl_scaling, smpl_trans

    @classmethod
    def load_frames(cls, video_path, frame_ids=None, fps=-1):
        """Load a single or multiple frames from a video."""
        if frame_ids is None:
            frame_ids = range(1e6)
        assert isinstance(frame_ids, list)
        if not os.path.exists(video_path):
            return None
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        assert cap.isOpened(), "check if your opencv is installed with ffmpeg supported."

        images = []
        for frame_id in frame_ids:
            sec = frame_id * 1.0 / fps
            cap.set(cv2.CAP_PROP_POS_MSEC, (sec * 1000))
            success, image = cap.read()
            if not success:
                break
            images.append(image)

        if len(images) > 0:
            images = np.stack(images)       # N, 3, H, W
        else:
            images = None

        cap.release()
        return images
