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
import argparse


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


def evaluate_model(args):
    frame_ids = [0, 's']
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    scales = [0, 1, 2, 3]
    STEREO_SCALE_FACTOR = 5.4
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80


    height = args.height
    width = args.width
    batch_size = args.batch_size

    device = torch.device(args.device)

    total_mean_errors = []

    num_ch_enc = np.array([64, 64, 128, 256, 512])
    sizes = (batch_size, height, width)

    ccnext_enc = models.ConvnextCAEncoder(sizes, device, args.window_size)
    idep = models.IDEP_Skip_Dual(np.array([32, 64, 128, 256, 512]))

    ccnext_enc.load_state_dict(torch.load(f"{args.model_path}/{args.encoder_path}").to("cpu"))
    idep.load_state_dict(torch.load(f"{args.model_path}/{args.decoder_path}").to("cpu"))

    ccnext_enc.eval()
    idep.eval()

    ccnext_enc.to(device)
    idep.to(device)


    test_filenames = readlines(args.filenames)
    dataset = datasets.KITTIRAWDataset(args.dataset_path, test_filenames, height, width, frame_ids, 4, is_train=False, img_ext=".jpg")

    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    num_layers = args.num_layers
    min_depth = args.min_depth
    max_depth = args.max_depth
    
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

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()
            input_color_s = data[("color", "s", 0)].cuda()

            encoder_out = ccnext_enc(input_color, input_color_s)
            output = idep(encoder_out)

            pred_disp, _ = disp_to_depth(output[('disp', 0)], min_depth, max_depth)

            pred_disp = pred_disp.cpu()[:, 0].numpy()
            # N = 1
            # pred_disp = pred_disp[:N]

            pred_disps.append(pred_disp[:batch_size])

    pred_disps = np.concatenate(pred_disps)


    gt_depths = np.load(args.gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    eval_split = args.eval_split
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

        pred_depth *= STEREO_SCALE_FACTOR

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)

    print(mean_errors)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate KITTI model')
    parser.add_argument('--encoder_path', type=str, default="encoder.pth", help='Path to the encoder model')
    parser.add_argument('--decoder_path', type=str, default="decoder.pth", help='Path to the decoder model')

    parser.add_argument("--window-size", type=float, help="Window Size for Cross Attention", default=0.26)
    parser.add_argument('--filenames', type=str, required=True, help='Path to the testing filenames')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the KITTI dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--height', type=int, default=192*2, help='Image height')
    parser.add_argument('--width', type=int, default=640*2, help='Image width')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--min_depth', type=float, default=0.1, help='Minimum depth')
    parser.add_argument('--max_depth', type=float, default=100, help='Maximum depth')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth depths')
    parser.add_argument('--eval_split', type=str, default='eigen_gargcrop', help='Evaluation split')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    evaluate_model(args)

if __name__ == "__main__":
    main()