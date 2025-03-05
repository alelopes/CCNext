import torch
import numpy as np
import glob
from utils import readlines
import datasets
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import einops
from layers import *
from utils import *
from kitti_utils import *
from torchvision import transforms
import cv2
import pandas as pd
from enum import Enum

import models

frame_ids = [0, 's']
side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
scales = [0, 1, 2, 3]
height = 192*2
width = 640*2
batch_size = 1

print("BS", batch_size)

filenames=""
dataset_path=""
model_path=""

device = torch.device('cuda:0')

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


total_mean_errors = []

num_ch_enc = np.array([64, 64, 128, 256, 512])
sizes = (batch_size, height, width)
device="cuda:0"

ccnext_enc = models.ConvnextCAEncoder(sizes, device, 0.26)
idep = models.IDEP_Skip_Dual(np.array([32, 64, 128, 256, 512]))

paths = "models_3/weights_6/"
ccnext_enc.load_state_dict(torch.load(f"{paths}/encoder.pth"))
idep.load_state_dict(torch.load(f"{paths}/decoder.pth"))

ccnext_enc.eval()
idep.eval()

ccnext_enc.to("cuda:0")
idep.to("cuda:0")

evaluate_dual = False

frame_ids = [0, 's']
side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
scales = [0, 1, 2, 3]
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 80

test_filenames = readlines("splits/eigen/test_files.txt")
#kitti_raw = "/hdd2/doutorado/car_datasets/KITTI/raw"
kitti_raw = '/work/souza_lab/alex/home/alexandre.lopes1/raw'
dataset = datasets.KITTIRAWDataset(kitti_raw, test_filenames, height, width, frame_ids, 4, is_train=False, img_ext=".jpg")

dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1,
                        pin_memory=True, drop_last=False)

num_layers = 18
min_depth = 0.1
max_depth = 100

# device = "cuda:0"
v1_multiscale = False
disable_automasking = False
no_ssim = False
predictive_mask = False
avg_reprojection = False

disparity_smoothness = 1e-3
num_scales = len(scales)

depth_metric_names = [
        "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]


backproject_depth = {}
project_3d = {}
for scale in scales:
    h = height // (2 ** scale)
    w = width // (2 ** scale)

    backproject_depth[scale] = BackprojectDepth(batch_size, h, w)
    backproject_depth[scale].to(device)

    project_3d[scale] = Project3D(batch_size, h, w)
project_3d[scale].to(device)

ssim = SSIM()
ssim.to(device)

pred_disps = []

print("dataloader", len(dataloader))

with torch.no_grad():
    for data in dataloader:
        input_color = data[("color", 0, 0)].cuda()
        input_color_s = data[("color", "s", 0)].cuda()

        encoder_out = ccnext_enc(input_color, input_color_s)
        output = idep(encoder_out)

        pred_disp, _ = disp_to_depth(output[('disp0', 0)], min_depth, max_depth)

        pred_disp = pred_disp.cpu()[:, 0].numpy()
        N = 1
        pred_disp = pred_disp[:N]

        pred_disps.append(pred_disp[:batch_size])


pred_disps = np.concatenate(pred_disps)

print("len pred disps", len(pred_disps))

disable_median_scaling = True
pred_depth_scale_factor = STEREO_SCALE_FACTOR

gt_path = os.path.join("../stereo_depth/splits/eigen/", "gt_depths.npz")
#    gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

#gt_path = os.path.join(<PATH FOR GT_DEPTH.NPZ FOLDER>, "gt_depths.npz")
gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

eval_split = "eigen_gargcrop"
errors = []
ratios = []

for i in range(pred_disps.shape[0]):

    gt_depth = gt_depths[i]
    gt_height, gt_width = gt_depth.shape[:2]

    pred_disp = pred_disps[i]
    pred_depth = 1 / pred_disp
    pred_depth = cv2.resize(pred_depth, (gt_width, gt_height), interpolation=cv2.INTER_LINEAR)

    if eval_split == "eigen_gargcrop":
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                        0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

    else:
        mask = gt_depth > 0

    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    pred_depth *= pred_depth_scale_factor

    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

    errors.append(compute_errors(gt_depth, pred_depth))

mean_errors = np.array(errors).mean(0)

print(mean_errors)

print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
print("\n-> Done!")



