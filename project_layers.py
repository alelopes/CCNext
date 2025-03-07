import torch
import torch.nn as nn
import torch.nn.functional as F


class BackprojectDepth(nn.Module):

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        grid_x, grid_y = torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing='xy')
        
        ones = torch.ones_like(grid_x)
        grid = torch.dstack((grid_x, grid_y, ones))
        
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        self.register_buffer('grid', grid)

        #grid is: batch, width, height, 3
        ones = torch.ones(batch_size, height, 1, width)
        self.register_buffer('ones', ones)


    def forward(self, depth, inv_K):
        # inv_K batch, 1, 4, 4
        inv_K = inv_K.unsqueeze(1)

        #depth is batch, 1, width, height
        depth = depth[:,0,:,:]
        #depth is batch, width, height
        depth = depth.unsqueeze(-1).repeat(1,1,1,3)
        # depth is batch, width, height, 3

        homogeneous_coord = self.grid*depth

        #batchx4x3 times batch x width x height x 3
        cam_coord = inv_K[:,:,:3,:3] @ homogeneous_coord.permute(0,1,3,2)

        #cam coords output is: batch x width x 3 x height
        cam_coord = torch.cat((cam_coord,self.ones), 2)
        #cam coords output is: batch x width x 4 x height
        return cam_coord#.permute(0,1,3,2)


class ProjectDepth(nn.Module):

    def __init__(self, height, width):
        super(ProjectDepth, self).__init__()
        self.height = height
        self.width = width
        #grid is: batch, width, height, 3


    def forward(self, world_coords, P):
        #world_coords shape is: batch x width x height x 4
        # P matrix is: batch_size x 4 x 4
        P = P.unsqueeze(1)
        
        homogeneous_coords = P @ world_coords

        pixel_coords = (homogeneous_coords/(homogeneous_coords[:,:,2:3,:]+1e-5))
        #homogeneous coord: batch x width x height x 4

        #convert to grid sample
        pixel_coords = pixel_coords.permute(0,1,3,2)
        pixel_coords = pixel_coords[...,:2]

        return pixel_coords
