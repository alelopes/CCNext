import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from layers import disp_to_depth, compute_depth_errors, get_smooth_loss
from utils import SSIM
from project_layers import BackprojectDepth, ProjectDepth
from models import ConvnextCAEncoder, IDEP_Skip_Dual

from datasets import kitti_dataset
from datasets import stereodrive_dataset

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()

MIN_DEPTH = 1e-3
MAX_DEPTH = 80
pred_depth_scale_factor = 5.4

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


class Trainer:
    def __init__(self, opt):
        super().__init__()

        #loading data    
        if opt.dataset == "KITTI":
            dataset = kitti_dataset.KITTIRAWDataset
        elif opt.dataset == "DrivingStereo":
            dataset = stereodrive_dataset.StereoDriveRAWDataset
        else:
            dataset = None

        train_filenames = readlines(opt.train_files)
        val_filenames = readlines(opt.val_files)
        frame_ids = [0, 's']
        scales = [0,1,2,3]
        ext = ".png" if opt.png else ".jpg"
        ext=".jpg"

        train_set = dataset(opt.data_path, train_filenames, opt.height, opt.width, frame_ids, 4, is_train=True, img_ext=ext)
        val_set = dataset(opt.data_path, val_filenames, opt.height, opt.width, frame_ids, 4, is_train=False, img_ext=ext)

        train_loader = DataLoader(train_set, opt.batch_size, True, num_workers=opt.workers, pin_memory=False, drop_last=True)
        val_loader = DataLoader(val_set, opt.batch_size, True, num_workers=opt.workers, pin_memory=False, drop_last=True)


        if opt.device>=0:
            device = torch.device(f"cuda:{opt.device}")
        else:
            device = torch.device("cpu")

        backproject = BackprojectDepth(opt.batch_size, opt.height, opt.width)
        backproject = backproject.to(device)
        self.backproject_depth = backproject

        project_depth = ProjectDepth(opt.height, opt.width)
        project_depth = project_depth.to(device)
        self.project_depth = project_depth
        self.ssim = SSIM()

        #loading model
        sizes = (opt.batch_size, opt.height, opt.width)

        ccnext_enc = ConvnextCAEncoder(sizes, device, opt.window_size)
        idep = IDEP_Skip_Dual(np.array([32, 64, 128, 256, 512]))

        ccnext_enc.to(device)
        idep.to(device)

        training_parameters = []
        training_parameters += list(ccnext_enc.parameters())
        training_parameters += list(idep.parameters())
        
        optim = torch.optim.Adam(training_parameters, opt.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, opt.lr_scheduler, 0.1)

        self.enc = ccnext_enc
        self.dec = idep
        
        self.optimizer = optim
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = opt.epochs
        self.device = device
        self.scales = scales
        self.width = opt.width
        self.height = opt.height
        self.disparity_smoothness = opt.disparity_smoothness
        self.save_path = opt.save_path

        self.depth_metric_names = [
                    "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]        


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        source_scale = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []
            reprojection_losses_2 = []

            disp = outputs[("disp", scale)]
            disp_2 = outputs[("disp_s", scale)]

            color = inputs[("color_aug", 0, scale)]
            color_sec = inputs[("color_aug","s", scale)]

            target = inputs[("color_aug", 0, source_scale)]
            target_sec = inputs[("color_aug", "s", source_scale)]

            pred = outputs[("color", 0, scale)]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            pred = outputs[("color", 's', scale)]
            reprojection_losses_2.append(self.compute_reprojection_loss(pred, target_sec))

            reprojection_losses = torch.cat(reprojection_losses, 1)
            reprojection_losses_2 = torch.cat(reprojection_losses_2, 1)

            identity_reprojection_losses = []
            identity_reprojection_losses_2 = []

            pred = inputs[("color_aug", 's', source_scale)]
            pred_2 = inputs[("color_aug", 0, source_scale)]
            identity_reprojection_losses.append(
                self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses_2.append(
                self.compute_reprojection_loss(pred_2, target_sec))


            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_losses_2 = torch.cat(identity_reprojection_losses_2, 1)

            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses
            identity_reprojection_loss_2 = identity_reprojection_losses_2

            reprojection_loss = reprojection_losses
            reprojection_loss_2 = reprojection_losses_2

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=self.device) * 0.00001

            identity_reprojection_loss_2 += torch.randn(
                identity_reprojection_loss_2.shape, device=self.device) * 0.00001                    

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            combined_2 = torch.cat((identity_reprojection_loss_2, reprojection_loss_2), dim=1)


            to_optimise, idxs = torch.min(combined, dim=1)
            to_optimise_2, idxs_2 = torch.min(combined_2, dim=1)

            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()
            outputs["identity_selection_2/{}".format(scale)] = (
                idxs_2 > identity_reprojection_loss_2.shape[1] - 1).float()

            loss += to_optimise.mean() #/ (2 ** 4)
            loss += to_optimise_2.mean() #/ (2 ** 4)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            mean_disp = disp_2.mean(2, True).mean(3, True)
            norm_disp = disp_2 / (mean_disp + 1e-7)
            smooth_loss_2 = get_smooth_loss(norm_disp, color_sec)

            loss += self.disparity_smoothness * smooth_loss
            loss += self.disparity_smoothness * smooth_loss_2

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= (len(self.scales)*2)
        losses["loss"] = total_loss
        return losses


    def image_from_coords(self, inpt, pixel_coords, width, height):
        # convert to grid sample
        pixel_coords[..., 0] /= width-1
        pixel_coords[..., 1] /= height-1
        pixel_coords *= 2
        pixel_coords -= 1  

        return F.grid_sample(
                    inpt, pixel_coords,
                    padding_mode='border'
                )        

    def images_from_depth(self, inputs, outputs):
        
        for scale in self.scales:
            disp = outputs[('disp', scale)]
            disp_s = outputs[('disp_s', scale)]

            disp = F.interpolate(
                disp, [self.height, self.width], mode="bilinear", align_corners=False)

            disp_s = F.interpolate(
                disp_s, [self.height, self.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, 0.1, 100)
            _, depth_s = disp_to_depth(disp_s, 0.1, 100)

            outputs['depth', 0, scale] = depth
            outputs['depth', 's', scale] = depth_s
            
            T = inputs["stereo_T"]
            K = inputs["K", 0]
            P = K @ T

            world_coords = self.backproject_depth(depth, inputs['inv_K', 0])
            pixel_coords_grid = self.project_depth(world_coords, P)

            outputs[("color", 0, scale)] = self.image_from_coords(inputs[("color_aug", "s", 0)], pixel_coords_grid, 
                                                                    self.width, self.height)

            T_sec = inputs["stereo_T"].clone()
            T_sec[:,0,3] = -1 * T_sec[:,0,3] #TO THE OTHER WAY AROUND
            K = inputs["K", 0]
            P = K @ T_sec

            world_coords = self.backproject_depth(depth_s, inputs['inv_K', 0])
            pixel_coords_grid = self.project_depth(world_coords, P)

            outputs[("color", "s", scale)] = self.image_from_coords(inputs[("color_aug", 0, 0)], pixel_coords_grid, 
                                                                    self.width, self.height)

    def train(self):
        print('starting training')
        self.enc.train()
        self.dec.train()
           
        with tqdm(self.train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {self.epoch}")

            for idx, batch in enumerate(tepoch):                    
                for key, val in batch.items():
                    batch[key] = val.to(self.device)
                self.optimizer.zero_grad()

                out_fist = self.enc(batch["color_aug", 0, 0], batch["color_aug", "s", 0])
                outputs = self.dec(out_fist)

                self.images_from_depth(batch, outputs)

                losses = self.compute_losses(batch, outputs)       

                losses["loss"].backward() 
                self.optimizer.step()

                tepoch.set_postfix(loss=losses["loss"].item())

                writer.add_scalar("Loss/train", losses["loss"].item(), len(self.train_loader)*self.epoch + idx)

        self.scheduler.step()




    def depth_metrics(self, inputs, pred_disp, is_print=False):

        _, _, h, w = inputs["depth_gt"].shape

        pred_depth = 1 / pred_disp.detach().cpu()
        pred_depth = F.interpolate(pred_depth, [h, w],
                         mode="bilinear", align_corners=False)

        pred_depth = pred_depth[:,0,:,:]

        gt_depth = inputs["depth_gt"].detach().cpu()[:,0,:,:]

        mask = torch.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * h, 0.99189189 * h,
                        0.03594771 * w,  0.96405229 * w]).astype(np.int32)

        crop_mask = torch.zeros_like(mask)
        crop_mask[:, crop[0]:crop[1], crop[2]:crop[3]] = 1

        mask = torch.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= pred_depth_scale_factor

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        if is_print:
            print("PRED", pred_depth)
            print("GT", gt_depth)

        depth_errors = compute_depth_errors(gt_depth, pred_depth)
        
        metrics = {}
        for i, metric in enumerate(self.depth_metric_names):
            metrics[metric] = np.array(depth_errors[i])

        return metrics
        

    def val(self):

        self.enc.eval()
        self.dec.eval()

        abs_rel_acc = 0
        idxs = 0
        abs_rel = 0
        sq_rel = 0
        rms = 0
        log_rms = 0
        a1 = 0
        a2 = 0
        a3 = 0

        with tqdm(self.val_loader, unit="batch") as tepoch:

            for idx, batch in enumerate(tepoch):                    
                for key, val in batch.items():
                    batch[key] = val.to(self.device)

                with torch.no_grad():
                    out_first = self.enc(batch["color", 0, 0], batch["color", "s", 0])
                    outputs = self.dec(out_first)

                self.images_from_depth(batch, outputs)

                losses = self.compute_losses(batch, outputs)       

                loss = losses["loss"].item() 
                tepoch.set_postfix(loss=losses["loss"].item())

                writer.add_scalar("Loss/eval", loss, len(self.val_loader)*self.epoch + idx)

                pred_disp, _ = disp_to_depth(outputs[('disp', 0)], 0.1, 100)

                if idx>=(len(self.val_loader)-3):
                    metrics = self.depth_metrics(batch, pred_disp, is_print=True)
                    print("abs rel", abs_rel_acc/idxs)
                else:
                    metrics = self.depth_metrics(batch, pred_disp, is_print=False)

                abs_rel_acc+=metrics["de/abs_rel"]
                sq_rel+=metrics["de/sq_rel"]
                rms+=metrics["de/rms"]
                log_rms+=metrics["de/log_rms"]
                a1+=metrics["da/a1"]
                a2+=metrics["da/a2"]
                a3+=metrics["da/a3"]

            idxs = len(self.val_loader)
            writer.add_scalar("val_abs_rel", abs_rel_acc/idxs, self.epoch)
            writer.add_scalar("val_sq_rel", sq_rel/idxs, self.epoch)
            writer.add_scalar("val_rms", rms/idxs, self.epoch)
            writer.add_scalar("val_log_rms", log_rms/idxs, self.epoch)
            writer.add_scalar("val_a1", a1/idxs, self.epoch)
            writer.add_scalar("val_a2", a2/idxs, self.epoch)
            writer.add_scalar("val_a3", a3/idxs, self.epoch)               


    def process(self):
                
        for epoch in range(self.epochs):
            self.epoch = epoch

            self.train()
            self.val()

            self.save_model()


    def save_model(self):
        """Save model weights to disk
        """


        save_folder = os.path.join(self.save_path, "weights_{}".format(self.epoch))
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, f"encoder.pth")
        to_save = self.enc.state_dict()
        torch.save(to_save, save_path)
        
        save_path = os.path.join(save_folder, f"decoder.pth")
        to_save = self.dec.state_dict()
        torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "optmizer.pth")
        torch.save(self.optimizer.state_dict(), save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    list_of_devices = [-1, 0, 1, 2, 3]
    list_of_datasets = ["KITTI", "DrivingStereo"]

    parser.add_argument('--device', type=int, help='cuda device, i.e. 0 or 0,1,2,3 or cpu (-1)', default=0, choices=list_of_devices)
    parser.add_argument('--data-path', type=str, default='yolo7.pt', help='Your Dataset Path')
    parser.add_argument('--dataset', type=str, default='KITTI', help='Your Training Dataset', choices=list_of_datasets)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')

    parser.add_argument("--height", type=int, help="Training height", default=384)
    parser.add_argument("--width", type=int, help="Training width", default=1280)
    parser.add_argument("--workers", type=int, help="Dataloader Workers", default=8)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=1e-4)
    parser.add_argument("--lr-scheduler", type=int, help="Learning Rate Scheduler", default=15)
    parser.add_argument("--disparity-smoothness", type=float, help="Disparity Smoothness", default=1e-3)
    parser.add_argument("--window-size", type=float, help="Window Size for Cross Attention", default=0.26)
    
    parser.add_argument("--save-folder", type=str, help="Folder to save the file", default="out_weights/")
    parser.add_argument("--train-files", type=str, help="Training Filenames", default='/home/alexandre.lopes1/monodepth_train/train')
    parser.add_argument("--val-files", type=str, help="Validation Filenames", default='/home/alexandre.lopes1/monodepth_train/val')
    parser.add_argument("--png", help="if data is png, activate --png", action="store_true")    
    parser.add_argument("--save-path", type=str, help="Path to save models", default="models")

    return parser.parse_args()




if __name__ == '__main__':
    opt = parse_args()
    
    trainer = Trainer(opt)
    trainer.process()
