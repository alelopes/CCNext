# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import cv2



def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoStereoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    Args:
        data_path
        depth_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 depth_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 is_test=False):
        super(MonoStereoDataset, self).__init__()

        self.data_path = data_path
        self.depth_path = depth_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.is_test = is_test
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            self.jitter_params = {'brightness': (.8, 1.2), 'contrast': (.8, 1.2), 'saturation': (.8, 1.2),
                                  'hue': (-.1, .1)}
            transforms.ColorJitter.get_params(*self.jitter_params.values())
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug, multi_time=False):
        """Resize colour images to the required scales and augment if required
        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
        if isinstance(color_aug, transforms.transforms.ColorJitter):
            for k in list(inputs):
                f = inputs[k]
                if "color" in k and k[1]==0 and not multi_time:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
                    k_s = list(k)
                    k_s[1] ='s'
                    k_s = tuple(k_s)
                    f_s = inputs[k_s]

                    inputs[(n, "s", i)] = self.to_tensor(f_s)

                    concat_features = torch.cat([self.to_tensor(f).unsqueeze(dim=0), self.to_tensor(f_s).unsqueeze(dim=0)], dim=0)
                    f_aug, f_s_aug = color_aug(concat_features)
                    
                    inputs[(n + "_aug", im, i)] = f_aug
                    inputs[(n + "_aug", "s", i)] = f_s_aug
                elif "color" in k and k[1]==0 and multi_time:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
                    k_s = list(k)
                    k_s[1] ='s'
                    k_s = tuple(k_s)
                    f_s = inputs[k_s]

                    k_minusone = list(k)
                    k_minusone[1] = -1
                    k_minusone = tuple(k_minusone)
                    f_minusone = inputs[k_minusone]

                    k_plusone = list(k)
                    k_plusone[1] = 1
                    k_plusone = tuple(k_plusone)
                    f_plusone = inputs[k_plusone]

                    inputs[(n, "s", i)] = self.to_tensor(f_s)
                    inputs[(n, -1, i)] = self.to_tensor(f_minusone)
                    inputs[(n, 1, i)] = self.to_tensor(f_plusone)

                    concat_features = torch.cat([self.to_tensor(f).unsqueeze(dim=0), self.to_tensor(f_s).unsqueeze(dim=0), self.to_tensor(f_minusone).unsqueeze(dim=0), self.to_tensor(f_plusone).unsqueeze(dim=0)], dim=0)
                    f_aug, f_s_aug, f_minusone, f_plusone = color_aug(concat_features)
                    
                    inputs[(n + "_aug", im, i)] = f_aug
                    inputs[(n + "_aug", "s", i)] = f_s_aug
                    inputs[(n + "_aug", -1, i)] = f_minusone
                    inputs[(n + "_aug", 1, i)] = f_plusone   
                # elif "color" in k and k[1]==-1 and k[2]==-1:
                #     inputs[("color_aug", -1, -1)] = self.to_tensor(f)                    
                #     inputs[("color_aug", 1, -1)] = self.to_tensor(f)                    


        else:            

            for k in list(inputs):
                # print("values", k)
                f = inputs[k]
                if "color" in k:
                    n, im, i = k
                    inputs[(n, im, i)] = self.to_tensor(f)
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

        # for k in list(inputs):
        #     print("values", k)
        #     f = inputs[k]
        #     if "color" in k and k[1]==0:
        #         n, im, i = k
        #         inputs[(n, im, i)] = self.to_tensor(f)
        #         k_s = list(k)
        #         k_s[1] ='s'
        #         k_s = tuple(k_s)
        #         f_s = inputs[k_s]

        #         inputs[(n, "s", i)] = self.to_tensor(f_s)

        #         f_aug, f_s_aug = color_aug([f, f_s])
                
        #         inputs[(n + "_aug", im, i)] = self.to_tensor(f_aug)
        #         inputs[(n + "_aug", "s", i)] = self.to_tensor(f_s_aug)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.
        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.
        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.01
        do_flip = self.is_train and random.random() > 0.5

        inputs["do_flip"] = do_flip

        # print("will flip?", do_flip)
        # print("doing color aug?", do_color_aug)

        filename = self.filenames[index]

        side = "l"

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color("", filename, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color("", filename, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(**self.jitter_params)
        else:
            color_aug = (lambda x: x)

        multi_time=False
        if len(self.frame_idxs)>2:
            multi_time=True

        self.preprocess(inputs, color_aug, multi_time)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(self.depth_path, filename, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
            
            if self.is_train:
                inputs["depth_gt"] = transforms.Resize((800, 1762))(inputs["depth_gt"])
            elif self.is_test:
                inputs["depth_gt"] = inputs["depth_gt"] #This is only intended for test phase. 
            else:
                inputs["depth_gt"] = transforms.Resize((800, 1762))(inputs["depth_gt"])


        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
