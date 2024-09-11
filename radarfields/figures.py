from copy import deepcopy

from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils.vis import arrays_to_gif
from utils.train import get_rotation_matrix

def save_polar(pred_fft, fft, data, fig_path, clip=True):
    """
    ## Save a side-by-side comparison of predicted and actual FFT\
    images, in polar coordinates
    """
    B, _, _ = pred_fft.shape

    timestamps = data["timestamps"]

    for b in range(B):

        fig, ax = plt.subplots(1,2)

        pred_fft = pred_fft.detach().cpu().numpy()
        fft = fft.detach().cpu().numpy()

        if clip: pred_fft = torch.clip(pred_fft, 0.0, 1.0)

        ax[0].set_title('Predicted FFT')
        ax[0].imshow(pred_fft)
        ax[1].set_title('FFT')
        ax[1].imshow(fft)

        fig.canvas.draw()
        fig.savefig(fig_path / f"{timestamps[b]}.png", dpi=300)
        plt.close()

def render_outputs_supersampled(trainer, data, points, bin_ranges, fig_path, timestamps, resolution=781):
    """## Query neural field with super-sampled cones to reconstruct predicted signal & render"""
    if resolution % 2 == 0: resolution += 1 # Resolution must be odd

    origins = points["origins"]  # [B, N*S, 3]
    B, _, _ = origins.shape
    S = data["num_fov_samples"]
    min_bin, max_bin = bin_ranges
    poses = data["poses"]
    fov_h = trainer.args.intrinsics_radar["opening_h"] # NOTE: these are in degrees
    fov_v = trainer.args.intrinsics_radar["opening_v"]
    radius = int(np.floor(resolution/2.0)) # image radius in pixels
    bin_size = trainer.args.intrinsics_radar["bin_size_radar"]

    # Computing correct pixel pitch to visualize all bins w/out aliasing
    min_radial_distance = (bin_size*min_bin)-(bin_size*0.5)
    max_radial_distance = (bin_size*max_bin)-(bin_size*0.5)
    pitch = max_radial_distance/radius

    # Matrix of image indices per pixel
    relative_indices_x = np.arange(-1*radius,radius+1) # [R = radius*2 + 1 = resolution]
    relative_indices_y = relative_indices_x # [R]
    img_indices = [[x, y] for y in relative_indices_y for x in relative_indices_x] # [R**2, 2] cartesian product

    # Converting matrix to tensor of radar coord points per pixel
    # (NOTE: z=0 is the mounting height of the radar, in radar coords)
    radar = torch.cat([torch.tensor(img_indices)*pitch, torch.zeros((resolution**2,1))], dim=-1) # [R**2, 3]
    radar = radar.to(trainer.device)

    # Masking out pixels that correspond to bins outside of the given bin range
    dists = torch.sqrt(torch.pow(radar[:,0], 2) + torch.pow(radar[:,1], 2)) # [R**2]
    mask = torch.logical_and(dists >= min_radial_distance, dists <= max_radial_distance) # [R**2]
    dists_np = dists.clone().detach().cpu().numpy()

    # Color map
    cm = plt.get_cmap("hot")
    cm_rd = plt.get_cmap("viridis")

    if S > 1: # Uniformly sample S*R**2 pitch-yaw pairs within radar opening angles
        dists = dists.repeat_interleave(S, dim=0) # [(R**2)*S]

        # Convert each radar point to a (roll, pitch, yaw) vector
        norms = torch.norm(radar, dim=1, keepdim=True) # [R**2, 1]
        unit_radar = radar / norms # [R**2, 3]
        yaw = torch.atan2(unit_radar[:, 1], unit_radar[:, 0])
        pitch = torch.zeros_like(yaw)
        roll = torch.zeros_like(pitch)
        rpy = torch.stack((roll, pitch, torch.rad2deg(yaw)), dim=1) # [R**2, 3]
        rpy = rpy.repeat_interleave(S, dim=0) # [(R**2)*S, 3]

        # Computing maximum elevation angle (pitch) for each of the R**2 sampled points
        # (NOTE: the max. pitch at any range R is the largest angle at which the 
        # corresponding ray would not cross the ground plane until distance R from the sensor)
        max_elev = torch.rad2deg(torch.atan2(torch.full_like(dists,1.62,device=trainer.device),dists))[..., None] # [(R**2)*S, 1]

        fov_samples_pitch = torch.linspace(0.0, 1.0, S, device=trainer.device)[None, :, None] # [1, S, 1]
        fov_samples_pitch = fov_samples_pitch * (max_elev.reshape((resolution**2, S, 1)) + (fov_v*0.5)) - (fov_v*0.5) # [R**2, S, 1]
        fov_samples_pitch = fov_samples_pitch.reshape((S*(resolution**2), 1)) # [(R**2)*S, 1]
        fov_samples_yaw = (torch.linspace(0.0, 1.0, S, device=trainer.device) - 0.5) * fov_h # [S]
        fov_samples_yaw = fov_samples_yaw[None, ...].expand((resolution**2, S)) # [R**2, S]
        fov_samples_yaw = fov_samples_yaw.reshape((S*(resolution**2)))[..., None] # [(R**2)*S, 1]
        fov_samples_roll = torch.zeros_like(fov_samples_pitch, device=trainer.device)
        fov_samples = torch.cat([fov_samples_roll, fov_samples_pitch, fov_samples_yaw], dim=-1) # [(R**2)S, 3]
        fov_samples[0:-1:S, :] = 0.0  # Set every Sth fov sample to 0,0,0, to preserve central ray

        # Sorting super-samples by elevation offset
        fov_samples = fov_samples.reshape((resolution**2, S, 3)) # [R**2, S, 3]
        fov_samples_pitch = fov_samples_pitch.reshape((resolution**2, S, 1))
        sorted_indices = torch.argsort(fov_samples_pitch, dim=1, descending=True)
        fov_samples = torch.gather(fov_samples, 1, sorted_indices.expand((resolution**2, S, 3)))
        fov_samples = fov_samples.reshape((S*(resolution**2), 3))

        rpy = rpy + fov_samples # [(R**2)S, 3]
        angular_offsets = fov_samples.cpu()
        rot_mats = get_rotation_matrix(rpy, roll_zero=True, device=trainer.device) # [(R**2)S, 3, 3]

        # Apply rotation matrices to forward-facing unit vectors (in radar space)
        forward = torch.tensor([1.0,0.0,0.0], device=trainer.device).type(torch.float32)
        forward = forward[None, :, None].expand((S*(resolution**2), 3, 1))
        rays_d = torch.bmm(rot_mats, forward) # [(R**2)S, 3, 1]

        # Rescale unit direction vectors by dists to recover radar coords
        radar = rays_d.squeeze(dim=-1) * dists[..., None] # [(R**2)S, 3]
    else: angular_offsets = None

    batch_indices = data["indices"]
    for b in range(B):

        data["indices"] = torch.tensor([batch_indices[b]])

        world_center = origins[b, 0, :] # [3]
        pose = poses[b, :, :] # [4, 4]
        timestamp = str(timestamps[b])

        # Converting points from radar to world coords
        radars = torch.cat((radar, torch.ones((S*(resolution**2), 1), device=trainer.device)), dim=-1) # [S*(R**2), 4] homogenous coords
        worlds = torch.einsum('rkl,rl->rk',pose[None,:,:].expand((S*(resolution**2), 4, 4)), radars)[:,:3] # [S*(R**2), 3]

        # Getting tensor of world coord direction vectors per pixel
        directions = worlds - world_center.expand_as(worlds) # [S*(R**2), 3]
        directions = F.normalize(directions, dim=1)

        # Computing white mask for corners
        white_mask = dists_np > max_radial_distance
        white_mask = white_mask.reshape((resolution, resolution))
        white_mask_fft = white_mask.copy()
        white_mask = np.tile(white_mask[..., None], 4)

        # Querying model
        query = deepcopy(points)
        query.update({
            "xyz": worlds,
            "directions": directions,
            "angular_offsets": angular_offsets
        })
        ranges_temp = data["ranges"]
        data["ranges"] = dists
        out = trainer.predict_waveform(data, query, batch=False)
        data["ranges"] = ranges_temp

        fft_img = (out[0]*mask).reshape((resolution, resolution)) # [R, R]
        fft_img = torch.clip(fft_img, 0.0, 1.0)
        alpha_img = out[1]
        alpha_img = alpha_img - torch.min(alpha_img[mask])
        alpha_img = alpha_img / torch.max(alpha_img[mask])
        alpha_img = (alpha_img*mask).reshape((resolution, resolution)) # [R, R]
        alpha_img = torch.clip(alpha_img, 0.0, 1.0)
        rd_img = out[2]
        rd_img = rd_img - torch.min(rd_img[mask])
        rd_img = rd_img / torch.max(rd_img[mask])
        rd_img = (rd_img*mask).reshape((resolution, resolution)) # [R, R]
        rd_img = torch.clip(rd_img, 0.0, 1.0)

        fft_img_path = fig_path / "pred_FFT"
        fft_img_path.mkdir(exist_ok=True)
        fft_img_path = fft_img_path / timestamp
        alpha_img_path = fig_path / "pred_occupancy"
        alpha_img_path.mkdir(exist_ok=True)
        alpha_img_path = alpha_img_path / timestamp
        rd_img_path = fig_path / "pred_reflectance"
        rd_img_path.mkdir(exist_ok=True)
        rd_img_path = rd_img_path / timestamp

        # Predicted FFT
        fft_img = fft_img.clone().detach().cpu().numpy()
        fft_img[white_mask_fft] = 1.0
        fft_img = Image.fromarray((fft_img*255).astype(np.uint8))
        fft_img.save(fft_img_path, compress_level=0)

        # Predicted occupancy
        alpha_img = cm(alpha_img.detach().cpu().numpy())
        alpha_img[white_mask] = 1.0
        alpha_img = Image.fromarray((alpha_img[:,:,:3]*255).astype(np.uint8))
        alpha_img.save(alpha_img_path, compress_level=0)

        # Predicted reflectance
        rd_img = cm_rd(rd_img.detach().cpu().numpy())
        rd_img[white_mask] = 1.0
        rd_img = Image.fromarray((rd_img[:,:,:3]*255).astype(np.uint8))
        rd_img.save(rd_img_path, compress_level=0)
    data["indices"] = batch_indices

def render_outputs_over_depth(trainer, data, points, bin_ranges, fig_path, timestamps,
                              z_bounds=(-1.3,1.3), step_size_z=0.05, resolution=781, gif=True):
    """
    ## Query model with 3D grid of points, render integrated measurements and\
    depth slices to visualize 3D neural field
    """
    if resolution % 2 == 0: resolution += 1 # resolution must be odd

    min_bin, max_bin = bin_ranges # range of bins at which to query model
    radius = int(np.floor(resolution/2.0)) # img radius in pixels
    poses = data["poses"]
    origins = points["origins"]  # [B, N*S, 3]
    B, NS, _ = origins.shape

    # Computing min. pixel pitch to visualize all bins without aliasing
    bin_size = trainer.args.intrinsics_radar["bin_size_radar"]
    min_radial_distance = (bin_size*min_bin)-(bin_size*0.5)
    max_radial_distance = (bin_size*max_bin)-(bin_size*0.5)
    pitch = max_radial_distance/radius

    # Z-axis slices at which to sample model
    z_levels = torch.arange(z_bounds[0], z_bounds[1]+step_size_z, step_size_z).to(trainer.device) # [Z]
    num_z_levels = len(z_levels)

    # Matrix of image indices per-pixel
    relative_indices_x = np.arange(-1*radius,radius+1) # [R = radius*2 + 1 = resolution]
    relative_indices_y = relative_indices_x[::-1] # [R]
    img_indices = [[x, y] for y in relative_indices_y for x in relative_indices_x] # [R**2, 2] cartesian product
    img_indices = torch.tensor(img_indices).to(trainer.device)*pitch

    # Adding z-dimension
    z_levels = z_levels[:, None, None].expand((num_z_levels, resolution**2, 1))
    img_indices = img_indices[None].expand((num_z_levels, resolution**2, 2))
    radar = torch.cat([img_indices, z_levels], dim=-1) # [Z, R**2, 3]

    # Getting elevation angle of each point
    dists = torch.sqrt(torch.pow(radar[...,0], 2) + torch.pow(radar[...,1], 2)) # [Z, R**2]
    elev_angles = torch.rad2deg(torch.atan2(z_levels[...,0], dists)).reshape((num_z_levels*resolution**2)) # [Z*(R**2)]
    dists_np = dists.clone().detach().cpu().numpy()
    elev_angles = elev_angles.detach().cpu().numpy()
    radar = radar.reshape((-1, 3)) # [Z*(R**2), 3]
    dists = dists.reshape((num_z_levels*(resolution**2)))# [Z*(R**2)]

    # Elevation radiation profile of sensor
    elev_LUT = data["elev_LUT"]
    elev_weights = np.interp(elev_angles, elev_LUT[:,0], elev_LUT[:,1])
    elev_weights = torch.from_numpy(elev_weights).type(torch.float32).to(trainer.device) # [Z*(R**2)]
    elev_weights = elev_weights.reshape((num_z_levels, resolution, resolution)) # [Z, R, R]
    total_weights_per_bin = torch.sum(elev_weights, dim=0) # [R, R]

    # Masking out pixels that correspond to bins outside of the given bin range
    mask = torch.logical_and(dists >= min_radial_distance, dists <= max_radial_distance)# [Z*(R**2)]
    mask = mask.reshape((num_z_levels, resolution, resolution)) # [Z, R, R]
    white_mask = dists_np > max_radial_distance # [Z, R**2]
    white_mask = white_mask.reshape((num_z_levels, resolution, resolution)) # [Z, R, R]
    white_mask = np.tile(white_mask[..., None], 4) # [Z, R, R, 4]

    # Color map
    cm = plt.get_cmap("hot")
    cm_rd = plt.get_cmap("viridis")

    batch_indices = data["indices"]
    for b in range(B):
        data["indices"] = torch.tensor([batch_indices[b]])

        world_center = origins[b, 0, :] # [3]
        pose = poses[b, :, :] # [4, 4]
        timestamp = str(timestamps[b])

        # Converting points from radar to world coords
        radars = torch.cat((radar, torch.ones((num_z_levels*(resolution**2), 1), device=trainer.device)), dim=-1) # [Z*(R**2), 4] homogenous coords
        worlds = torch.einsum('rkl,rl->rk',pose[None,:,:].expand((num_z_levels*(resolution**2), 4, 4)), radars)[:,:3] # [Z*(R**2), 3]

        # Getting tensor of world coord direction vectors per point
        directions = worlds - world_center.expand_as(worlds) # [Z*(R**2), 3]
        directions = F.normalize(directions, dim=1)

        # Querying model
        query = deepcopy(points)
        query.update({
            "xyz": worlds,
            "directions": directions
        })
        ranges_temp = data["ranges"]
        data["ranges"] = dists
        S_temp = data["num_fov_samples"]
        data["num_fov_samples"] = 1
        outs = trainer.predict_waveform(data, query, batch=False)
        data["ranges"] = ranges_temp
        data["num_fov_samples"] = S_temp

        # Occupancy and reflectance
        alpha_imgs = outs[1] # [Z*(R**2)]
        rd_imgs = outs[2]  # [Z*(R**2)]
        alpha_imgs = alpha_imgs.reshape((num_z_levels, resolution, resolution))
        rd_imgs = rd_imgs.reshape((num_z_levels, resolution, resolution))

        # Integrating across elevation slices using sensor elevation profile
        alpha_integrated = torch.sum(alpha_imgs*elev_weights, dim=0)/total_weights_per_bin # [R, R]
        rd_integrated = torch.sum(rd_imgs*elev_weights, dim=0)/total_weights_per_bin # [R, R]

        # Taking z-wise mean intensity per x-y position
        alpha_mean = torch.mean(alpha_imgs, dim=0) # [R, R]
        rd_mean = torch.mean(rd_imgs, dim=0) # [R, R]

        # Normalizing
        alpha_integrated = alpha_integrated - torch.min(alpha_integrated[mask[0,...]])
        alpha_integrated = alpha_integrated / torch.max(alpha_integrated[mask[0,...]])
        rd_integrated = rd_integrated - torch.min(rd_integrated[mask[0,...]])
        rd_integrated = rd_integrated / torch.max(rd_integrated[mask[0,...]])
        alpha_mean = alpha_mean - torch.min(alpha_mean[mask[0,...]])
        alpha_mean = alpha_mean / torch.max(alpha_mean[mask[0,...]])
        rd_mean = rd_mean - torch.min(rd_mean[mask[0,...]])
        rd_mean = rd_mean / torch.max(rd_mean[mask[0,...]])

        # Applying mask & clipping
        alpha_integrated = torch.clip(alpha_integrated*mask[0,...], 0.0, 1.0)
        rd_integrated = torch.clip(rd_integrated*mask[0,...], 0.0, 1.0)
        alpha_mean = torch.clip(alpha_mean*mask[0,...], 0.0, 1.0)
        rd_mean = torch.clip(rd_mean*mask[0,...], 0.0, 1.0)

        alpha_img_path = fig_path / "integrated_occupancy"
        alpha_img_path.mkdir(exist_ok=True)
        alpha_img_path = alpha_img_path / timestamp
        rd_img_path = fig_path / "integrated_reflectance"
        rd_img_path.mkdir(exist_ok=True)
        rd_img_path = rd_img_path / timestamp
        alpha_mean_img_path = fig_path / "mean_occupancy"
        alpha_mean_img_path.mkdir(exist_ok=True)
        alpha_mean_img_path = alpha_mean_img_path / timestamp
        rd_mean_img_path = fig_path / "mean_reflectance"
        rd_mean_img_path.mkdir(exist_ok=True)
        rd_mean_img_path = rd_mean_img_path / timestamp

        # Integrated occupancy
        alpha_integrated = cm(alpha_integrated.detach().cpu().numpy())
        alpha_integrated[white_mask[0,...]] = 1.0
        alpha_integrated = Image.fromarray((alpha_integrated[:,:,:3]*255).astype(np.uint8))
        alpha_integrated.save(alpha_img_path, compress_level=0)

        # Integrated reflectance
        rd_integrated = cm_rd(rd_integrated.detach().cpu().numpy())
        rd_integrated[white_mask[0,...]] = 1.0
        rd_integrated = Image.fromarray((rd_integrated[:,:,:3]*255).astype(np.uint8))
        rd_integrated.save(rd_img_path, compress_level=0)

        # Averaged occupancy
        alpha_mean = cm(alpha_mean.detach().cpu().numpy())
        alpha_mean[white_mask[0,...]] = 1.0
        alpha_mean = Image.fromarray((alpha_mean[:,:,:3]*255).astype(np.uint8))
        alpha_mean.save(alpha_mean_img_path, compress_level=0)

        # Averaged reflectance
        rd_mean = cm_rd(rd_mean.detach().cpu().numpy())
        rd_mean[white_mask[0,...]] = 1.0
        rd_mean = Image.fromarray((rd_mean[:,:,:3]*255).astype(np.uint8))
        rd_mean.save(rd_mean_img_path, compress_level=0)

        if gif:
            alpha_imgs = alpha_imgs - torch.min(alpha_imgs[mask])
            alpha_imgs = alpha_imgs / torch.max(alpha_imgs[mask])
            rd_imgs = rd_imgs - torch.min(rd_imgs[mask])
            rd_imgs = rd_imgs / torch.max(rd_imgs[mask])
            alpha_imgs = torch.clip(alpha_imgs*mask, 0.0, 1.0)
            rd_imgs = torch.clip(rd_imgs*mask, 0.0, 1.0)

            alpha_gif_path = fig_path / "occupancy_height_slices"
            alpha_gif_path.mkdir(exist_ok=True)
            alpha_gif_path = alpha_gif_path / (timestamp.strip('.png') + '.gif')
            rd_gif_path = fig_path / "reflectance_height_slices"
            rd_gif_path.mkdir(exist_ok=True)
            rd_gif_path = rd_gif_path / (timestamp.strip('.png') + '.gif')

            # Occupancy gif
            alpha_imgs = alpha_imgs.detach().cpu().numpy()
            alpha_imgs = cm(alpha_imgs[:, ::-1, ...])
            alpha_imgs[white_mask] = 1.0
            arrays_to_gif(alpha_imgs[...,:3]*255, alpha_gif_path)

            # Reflectance gif
            rd_imgs = rd_imgs.detach().cpu().numpy()
            rd_imgs = cm_rd(rd_imgs[:, ::-1, ...])
            rd_imgs[white_mask] = 1.0
            arrays_to_gif(rd_imgs[...,:3]*255, rd_gif_path)
    data["indices"] = batch_indices

    return