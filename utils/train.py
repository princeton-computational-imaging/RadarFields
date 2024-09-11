import os
import random

import torch
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_rotation_matrix(rpy, device="cpu", numpy=False, roll_zero=False):
    '''
    ## Convert roll,pitch,yaw (in degrees) to rotation matrix
    NOTE: we multiply rpy by -1 because we are doing CW rotations instead of CCW
    '''
    assert(len(rpy.shape) == 2 and rpy.shape[1] == 3)
    BNS = rpy.shape[0]
    rpy_rad = torch.deg2rad(rpy*-1.0) # [BNS, 3]

    C_x = torch.zeros((BNS, 3, 3), device=device)
    C_y = C_x.clone()
    C_z = C_x.clone()

    C_x[:,0,0] = 1.0
    if not roll_zero:
        sin_roll = torch.sin(rpy_rad[...,0]) # [BNS]
        cos_roll = torch.cos(rpy_rad[...,0]) # [BNS]
        C_x[:,1,1] = cos_roll
        C_x[:,2,2] = cos_roll
        C_x[:,1,2] = -1.0*sin_roll
        C_x[:,2,1] = sin_roll
    else: # roll angle is always 0; avoid unnecessary computation
        # cos(0) == 1.0
        C_x[:,1,1] = 1.0
        C_x[:,2,2] = 1.0

    C_y[:,1,1] = 1.0
    sin_pitch = torch.sin(rpy_rad[...,1]) # [BNS]
    cos_pitch = torch.cos(rpy_rad[...,1]) # [BNS]
    C_y[:,0,0] = cos_pitch
    C_y[:,2,2] = cos_pitch
    C_y[:,2,0] = -1.0*sin_pitch
    C_y[:,0,2] = sin_pitch

    C_z[:,2,2] = 1.0
    sin_yaw = torch.sin(rpy_rad[...,2]) # [BNS]
    cos_yaw = torch.cos(rpy_rad[...,2]) # [BNS]
    C_z[:,0,0] = cos_yaw
    C_z[:,1,1] = cos_yaw
    C_z[:,0,1] = -1.0*sin_yaw
    C_z[:,1,0] = sin_yaw

    rot = torch.bmm(C_z, C_y)  # [BNS, 3, 3]
    rot = torch.bmm(rot, C_x) # [BNS, 3, 3]
    return (rot.numpy() if numpy else rot)

def range_to_world(ranges, bin_length):
    '''
    ## Convert range bin indices into world coords
    ### (radial distances in meters)
    '''
    return ranges * bin_length - (bin_length/2.0)