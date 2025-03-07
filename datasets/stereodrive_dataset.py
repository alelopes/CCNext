# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
# from .mono_stereodrive_correct_B import MonoStereoDataset, MonoStereoDatasetDisp
from .mono_stereodrive import MonoStereoDataset



class StereoDriveDataset(MonoStereoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(StereoDriveDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[1.13911, 0, 0.5, 0],
                           [0, 2.50889, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (881, 400)
        self.side_map = {"2": 2, "3": 3, "l": "train-left-image", "r": "train-right-image"}

    def check_depth(self):

        return True

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


# class StereoDriveDatasetDisp(MonoStereoDatasetDisp):
#     """Superclass for different types of KITTI dataset loaders
#     """
#     def __init__(self, *args, **kwargs):
#         super(StereoDriveDatasetDisp, self).__init__(*args, **kwargs)

#         # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
#         # To normalize you need to scale the first row by 1 / image_width and the second row
#         # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
#         # If your principal point is far from the center you might need to disable the horizontal
#         # flip augmentation.
#         self.K = np.array([[1.13911, 0, 0.5, 0],
#                            [0, 2.50889, 0.5, 0],
#                            [0, 0, 1, 0],
#                            [0, 0, 0, 1]], dtype=np.float32)

#         self.full_res_shape = (881, 400)
#         self.side_map = {"2": 2, "3": 3, "l": "train-left-image", "r": "train-right-image"}

#     def check_depth(self):

#         return True

#     def get_color(self, folder, frame_index, side, do_flip):
#         color = self.loader(self.get_image_path(folder, frame_index, side))

#         if do_flip:
#             color = color.transpose(pil.FLIP_LEFT_RIGHT)

#         return color


class StereoDriveRAWDataset(StereoDriveDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(StereoDriveRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, f_str, side):
        image_path = os.path.join(
            self.data_path, folder, self.side_map[side], f_str)
        return image_path

    def get_depth(self, folder, frame, side, do_flip):
        depth = cv2.imread(os.path.join(folder, frame[:-3] + "png"), -1)
        depth = depth.astype(np.float32) / 256

        return depth


# class StereoDriveRAWDatasetDisp(StereoDriveDatasetDisp):
#     """KITTI dataset which loads the original velodyne depth maps for ground truth
#     """
#     def __init__(self, *args, **kwargs):
#         super(StereoDriveRAWDatasetDisp, self).__init__(*args, **kwargs)

#     def get_image_path(self, folder, f_str, side):
#         image_path = os.path.join(
#             self.data_path, folder, self.side_map[side], f_str)
#         return image_path

#     def get_depth(self, folder, frame, side, do_flip):
#         depth = cv2.imread(os.path.join(folder, frame[:-3] + "png"), -1)
#         depth = depth.astype(np.float32) / 256

#         return depth

#     def get_disp(self, folder, frame, side, do_flip):
#         depth = cv2.imread(os.path.join(folder, frame[:-3] + "png"), -1)
#         depth = depth.astype(np.float32) / 256

#         return depth


class StereoDriveDepthDataset(StereoDriveDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(StereoDriveDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, f_str, side):
        image_path = os.path.join(
            self.data_path, folder, self.side_map[side], f_str)
        return image_path


    def get_depth(self, folder, frame_index, side, do_flip):
        depth = cv2.imread(os.path.join(folder, frame[:-3] + "png"), -1)
        depth = depth.astype(np.float32) / 256

        return depth
