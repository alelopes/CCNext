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
    MAX_DEPTH = 120

    height = args.height
    width = args.width
    batch_size = args.batch_size

    device = torch.device(args.device)

    total_mean_errors = []

    num_ch_enc = np.array([64, 64, 128, 256, 512])
    sizes = (batch_size, height, width)

    ccnext_enc = models.ConvnextCAEncoder(sizes, device, args.window_size)

    if args.reduced_decoder:
        idep = models.IDEP_Skip(np.array([32, 64, 128, 256, 512]))
    else:
        idep = models.IDEP_Skip_Dual(np.array([32, 64, 128, 256, 512]))

    ccnext_enc.load_state_dict(torch.load(f"{args.model_path}/{args.encoder_path}"))
    idep.load_state_dict(torch.load(f"{args.model_path}/{args.decoder_path}"))

    ccnext_enc.eval()
    idep.eval()

    ccnext_enc.to(device)
    idep.to(device)


    test_filenames = readlines(args.filenames)

    dataset = datasets.StereoDriveRAWDataset(
        os.path.join(args.dataset_path, args.drivingstereo_test_images),
        os.path.join(args.dataset_path, args.drivingstereo_test_depth),
        test_filenames, args.height, args.width, frame_ids, 4, is_train=False, img_ext=args.ext, is_test=True
    )

    print(f"loaded dataset with {len(dataset)} samples")

    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

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
    errors = []

    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()
            input_color_s = data[("color", "s", 0)].cuda()

            encoder_out = ccnext_enc(input_color, input_color_s)
            output = idep(encoder_out)

            pred_disp, pred_depth = disp_to_depth(output[('disp', 0)], min_depth, max_depth)

            pred_depth*=5.4

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # depth_pred = torch.clamp(F.interpolate(
            #     torch.tensor(pred_depth), [400, 881], mode="bilinear", align_corners=False), 1e-3, 120)
            depth_pred = torch.clamp(F.interpolate(
                torch.tensor(pred_depth), [800, 1762], mode="bilinear", align_corners=False), 1e-3, 120)
            
            # print("compare preds", depth_pred.shape, pred_depth.shape)

            depth_pred = depth_pred[:,0,:,:]
            depth_gt = data['depth_gt'].detach().cpu()[:,0,:,:]

            gt_height, gt_width = depth_gt.shape[-2:]
            
            # if right_pad>0:
            #     depth_gt = depth_gt[:, top_pad:,:-right_pad]
            # else:
            #     depth_gt = depth_gt[:, top_pad:,:]            
            
            if args.eval_split == "eigen_gargcrop":
                mask = np.logical_and(depth_gt > MIN_DEPTH, depth_gt < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[:, crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            
            else:
                mask = depth_gt > 0


            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]

            # print(compute_errors(depth_gt.cpu().numpy(), depth_pred.cpu().numpy()))
            errors.append(compute_errors(depth_gt.cpu().numpy(), depth_pred.cpu().numpy()))

    mean_errors = np.array(errors).mean(0)

    print(mean_errors)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate DrivingStereo model')
    parser.add_argument('--encoder-path', type=str, default="encoder.pth", help='Path to the encoder model')
    parser.add_argument('--decoder-path', type=str, default="decoder.pth", help='Path to the decoder model')

    parser.add_argument("--window-size", type=float, help="Window Size for Cross Attention", default=0.5)
    parser.add_argument('--filenames', type=str, required=True, help='Path to the testing filenames')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the DrivingStereo dataset')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model')
    # parser.add_argument('--batch-size', type=int, default=1, help='Batch size') #Batch Size will be 1 because the dimensions of the depth maps changes.
    parser.add_argument('--height', type=int, default=288, help='Image height')
    parser.add_argument('--width', type=int, default=640, help='Image width')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--min-depth', type=float, default=0.1, help='Minimum depth')
    parser.add_argument('--max-depth', type=float, default=100, help='Maximum depth')
    parser.add_argument('--eval-split', type=str, default='eigen_gargcrop', help='Evaluation split')
    parser.add_argument('--reduced-decoder', action='store_true', help='Use Single Decoder network instead of two Output Decoders')
    parser.add_argument('--ext', type=str, default='jpg', help='Image file extension (jpg or png)')

    parser.add_argument("--drivingstereo-test-images", required=True, type=str, help="Folder containing validation images for DrivingStereo", default='/path/to/test/images')
    parser.add_argument("--drivingstereo-test-depth", required=True, type=str, help="Folder containing validation depth maps for DrivingStereo", default='/path/to/test/depth')


    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    evaluate_model(args)

if __name__ == "__main__":
    main()