import torch

from utils.train import get_rotation_matrix

def get_azimuths(B, N, num_azimuths, device, all=False):
    '''
    ## Sample N different azimuth angles in [0,num_azimuths) for\
    each of the B radar frames in a batch
    '''
    if all: # Sample all azimuth angles of the radar
        azim_samples = torch.arange(0,num_azimuths, device=device)[None].expand((B,num_azimuths))
    else: azim_samples = torch.sort(torch.randint(0, num_azimuths, (B, N), device=device))[0]
    return azim_samples # [B, N]

def get_radar_rays(poses, intrinsics_radar, num_azimuths, N, num_fov_samples, azim_samples, device):
    '''
    ## Given radar poses & azimuth samples, sample world-coordinate ray origins and directions
    :param poses (torch.Tensor): radar2world matrices
    :param intrinsics_radar (tuple): azimuth & elevation fov
    :param num_azimuths (int): total number of azimuth beams
    :param N (int): number of azimuth samples
    :param num_fov_samples (int): number of cone super-samples per azimuth beam
    :param azim_samples (torch.Tensor): sampled azimuth angles
    # '''
    B = poses.shape[0] # [B, 4, 4]
    S = num_fov_samples # num of ray super-samples within radar beam
    fov_h, fov_v = intrinsics_radar #NOTE: these are in degrees

    rays = {}

    # Convert sampled azimuth angles to degrees & expand
    azim_degrees = (360.0/num_azimuths) * azim_samples # [B, N] NOTE: we rotate clockwise
    azim_degrees = azim_degrees.repeat_interleave(num_fov_samples, dim=1) # [B, N*S]

    # Convert azimuth samples into roll, pitch, yaw, with roll and pitch set to 0
    azim_rpy_degrees = torch.stack([torch.zeros_like(azim_degrees, device=device),
                                   torch.zeros_like(azim_degrees, device=device),
                                   azim_degrees], dim=2) # [B, N*S, 3]
    
    if S > 1: # Randomly sample B*N*S pitch-yaw offset pairs within radar opening angles
        fov_samples_pitch = (torch.rand((B, N*S, 1), device=device) - 0.5) * fov_v
        fov_samples_yaw = (torch.rand((B, N*S, 1), device=device) - 0.5) * fov_h
        fov_samples_roll = torch.zeros_like(fov_samples_pitch, device=device)
        fov_samples = torch.cat([fov_samples_roll, fov_samples_pitch, fov_samples_yaw], dim=2) # [B, N*S, 3]

        # Set every Sth sample to 0,0,0, to preserve central ray in the cone
        fov_samples[:, 0:-1:S, :] = 0.0 
        fov_samples_pitch[:, 0:-1:S, :] = 0.0

        # Sort offsets by pitch, descending (along z)
        fov_samples = fov_samples.reshape((B, N, S, 3)) # [B, N, S, 3]
        fov_samples_pitch = fov_samples_pitch.reshape((B, N, S, 1)) # [B, N, S, 1]
        sorted_indices = torch.argsort(fov_samples_pitch, dim=2, descending=False) # [B, N, S, 1]
        fov_samples = torch.gather(fov_samples, 2, sorted_indices.expand((B, N, S, 3))) # [B, N, S, 3]
        fov_samples = fov_samples.reshape((B, N*S, 3)) # [B, N*S, 3]

        # Apply sampled pitch-yaw pairs as offsets to sampled yaws (azimuth)
        azim_rpy_degrees = azim_rpy_degrees + fov_samples # [B, N*S, 3]
        rays["fov_samples"] = fov_samples.cpu()
    else:
        rays["fov_samples"] = None

    # compute [B, N*S] rotation matrices in radar space for each ray direction
    rot_mats = get_rotation_matrix(azim_rpy_degrees.reshape((B*N*S, 3)), device=device,
                                   roll_zero=True).reshape((B, N*S, 3, 3))

    # Apply rotation matrices to forward-facing unit vectors (in radar space)
    forward = torch.tensor([1.0,0.0,0.0], device=device).type(torch.float32)
    forward = forward[None, None, :].expand((B, N*S, 3))
    directions = torch.einsum('bnkl,bnl->bnk', rot_mats, forward) # [B, N*S, 3]

    # Use radar2world to map [B, N*S, 3] radar ray directions to world coords
    directions = torch.einsum('bnkl,bnl->bnk',poses[:, None, :3, :3].expand((B, N*S, 3, 3)),
                          directions.reshape([B, N*S,3]))
    rays["directions"] = directions # [B, N*S, 3]

    # NOTE: each bundle of S rays will share the same origin
    origins = torch.tensor([0.0,0.0,0.0,1.0], device=device)[None,None].expand((B, N*S, 4))
    origins = torch.einsum('bnkl,bnl->bnk', poses[:, None, :, :].expand((B, N*S, 4, 4)),
                          origins.type(torch.float32))
    rays["origins"] = origins[..., :3] # [B, N*S, 3]

    return rays

def get_points(data_batch, args, device):
    '''
    ## Convert a batch of range-azimuth bin samples\
    into cones of 3D points (in world coords) that can\
    be used to query a Radar Field
    '''
    points = {}

    B = data_batch["bs"]
    N = data_batch["num_rays_radar"] # num azimuths sampled
    S = data_batch["num_fov_samples"] # num cone super-samples
    R = data_batch["num_range_samples"]
    intrinsics = data_batch["intrinsics"]
    total_azimuths = args.intrinsics_radar["num_azimuths_radar"]
    poses = data_batch["poses"] # [B, 4, 4] radar2world matrices
    azimuths = data_batch["azimuths"] # [B, N] azimuth samples
    ranges = data_batch["ranges"] # [B, N*S, R] range bin samples

    # Given radar poses & samples, compute ray origins and directions
    rays = get_radar_rays(poses, intrinsics, total_azimuths,
        N, S, azimuths, device)

    # Sampled ray origin and direction vectors
    origins = rays["origins"] # [B, N*S, 3]
    directions = rays["directions"] # [B, N*S, 3]

    # Compute xyz queries from ray origins, directions, and ranges
    xyz = origins[:,:,None,:].expand((B, N*S, R, 3))
    directions = directions[:,:,None,:].expand((B, N*S, R, 3))
    xyz = xyz + (directions * ranges[...,None]) # [B, N*S, R, 3]

    points.update({
        "xyz": xyz,
        "directions": directions,
        "origins": origins
    })

    # NOTE: we need to know the angular offset of each super-sampler ray for
    # weighted beam integration w/ antenna radiation profile
    if args.integrate_rays:
        points["angular_offsets"] = rays["fov_samples"] # [B, N*S, 3]

    return points

def get_range_samples(B, N, num_samples, bounds, device, all=False):
    '''
    ## Randomly samples num_samples different ranges for each of the\
    B*N rays. Samples are integer values in [bounds[0],bounds[1]],\
    inclusive, which correspond to range bin indices in the radar FFT data.

    :param B (int): batch size
    :param N (int): number of azimuth rays per batch
    :param num_samples (int): number of range bins to sample
    :param bounds (tuple): [lower, upper] bounds on sampled indices
    :param all (bool): if True, return all indices within bounds
    :return samples (torch.Tensor): [B, N, R] range samples; R=num_samples
    '''
    min, max = bounds[0], bounds[1]
    assert(min > 0)
    num_bins = max-min+1
    
    if all: return torch.arange(min,max+1,device=device)[None,None].expand((B, N, num_bins))

    assert(num_samples <= num_bins)

    samples = torch.rand(B, num_bins, device=device).argsort(dim=-1)[:, :num_samples] + min
    samples, _ = torch.sort(samples, dim=-1)
    return samples[:, None, :].expand((B, N, num_samples))