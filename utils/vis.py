import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from PIL import Image
import imageio

def plot_trajectory(poses, path, size=0.05):
    """## Save 3D plot of sensor trajectory from GNSS poses"""
    num_poses = poses.shape[0]

    radar_center = torch.tensor([0.0, 0.0, 0.0], device=poses.device)
    radar_forward = torch.tensor([1.0, 0.0, 0.0], device=poses.device)
    radar_forward = radar_forward[None,:,None].expand((num_poses, 3, 1))

    # Mapping center and forward vector to world coords
    # for each frame using corresponding pose
    world_centers = radar_center + poses[:,:3,-1] # [num_poses, 3]
    world_forwards = torch.bmm(poses[:,:3,:3], radar_forward)[...,0] # [num_poses, 3]

    world_centers = world_centers.detach().cpu().numpy()
    world_forwards = world_forwards.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=0.0, azim=320.0)
    ax.dist = 6.0
    ax.scatter(world_centers[...,0], world_centers[...,1], world_centers[...,2], s=size*5)
    ax.quiver(world_centers[...,0], world_centers[...,1], world_centers[...,2],
              world_forwards[...,0], world_forwards[...,1], world_forwards[...,2],
              length=size*13)
    fig.canvas.draw()
    fig.savefig(path, dpi=300)
    plt.close()

def render_FFT(fft_img, num_bins_to_show, bin_size, num_azims, path,
                          resolution=781, noise_floor=None, save=True, plot=False, norm=False):
    """## Interpolate to render a single FFT frame in cartesian coordinates"""

    height, width = fft_img.shape
    assert(num_bins_to_show <= width)
    radians_per_azim = np.pi*2.0/num_azims

    # Setting up grid interpolator
    a_pts = np.arange(-1, num_azims+1)
    r_pts = np.arange(num_bins_to_show)
    fft_img_wrapped = np.zeros((height+2,width))
    fft_img_wrapped[1:height+1,:] = fft_img
    fft_img_wrapped[0,:] = fft_img[-1,:]
    fft_img_wrapped[-1,:] = fft_img[0,:]
    interp = RegularGridInterpolator((a_pts, r_pts), fft_img_wrapped[:,:num_bins_to_show],
                                     bounds_error=False, fill_value=0.0)

    render = np.zeros((resolution, resolution))
    white_mask = np.zeros_like(render)

    # Computing pixel pitch and range bounds for rendered img
    max_radial_distance = (num_bins_to_show)*bin_size - (bin_size/2.0)
    pitch = ((num_bins_to_show * 2.0) - 1) * bin_size / resolution

    # Array of image indices
    i, j = np.meshgrid(range(resolution), range(resolution-1,-1,-1), indexing='ij')
    img_coords = np.stack((i, j), axis=-1)*pitch-max_radial_distance #[resolution, resolution, 2]

    # Computing pixel world polar coords
    radial_distances = np.sqrt(img_coords[...,0]**2 + img_coords[...,1]**2)
    azimuths = np.arctan2(img_coords[...,1],img_coords[...,0])+(np.pi/2.0)
    azimuths[azimuths < 0.0] += 2*np.pi
    white_mask[radial_distances > max_radial_distance] = 1.0

    # Mapping to FFT pixel space
    rs = (radial_distances / bin_size) - 1
    azs = (azimuths / radians_per_azim) - 1
    render = interp(np.stack((azs, rs), axis=-1)) # [resolution, resolution]

    if norm:
        render = render - np.min(render)
        render = render / np.max(render)

    # Noise thresholding
    if noise_floor is not None:
        has_signal = render > noise_floor
        render = render * has_signal
    
    render[white_mask > 0.5] = 1.0

    if save:
        render = Image.fromarray((render*255).astype(np.uint8))
        render.save(path, compress_level=0)
    if plot:
        plt.imshow(render)
        plt.show()
    if not save and not plot: return render

def render_FFT_batch(fft_imgs, num_bins_to_show, bin_size, num_azims, path, timestamps, min_range_bin,
                     resolution=781, noise_floor=None, save=True, no_override=False):
    '''## Render a batch of FFT images, in cartesian coordinates'''
    min_bin = min_range_bin-1 # converting from 1-indexed to 0-indexed
    B, height, width_og = fft_imgs.shape
    width = width_og + min_bin
    assert(num_bins_to_show < width)

    for b in range(B):
        fpath = path / str(timestamps[b])
        if no_override:
            if fpath.is_file(): continue

        fft_img = fft_imgs[b,:,:].clone().detach().cpu().numpy()

        # Filling in empty center
        fft_img_temp = np.zeros((height, width))
        fft_img_temp[:,min_bin:] = fft_img
        fft_img = fft_img_temp

        render_FFT(fft_img, num_bins_to_show, bin_size, num_azims, fpath,
                              resolution=resolution, noise_floor=noise_floor, save=save)

def save_loss_plot(loss, img_dir, args, global_step):
    """## Save plot of all loss terms & regularization penalties over train time"""
    name = args.name.strip("\"")
    img_path = img_dir / f"{name}_loss-global_step{global_step}.png"

    fig = plt.figure()

    # Getting all keys from the loss dict
    # (each key corresponds to a line we will plot)
    losses = loss.keys()
    for loss_term in losses:
        if len(loss[loss_term]) < 1: continue
        plt.plot(loss[loss_term], label=loss_term)
    plt.xlabel("Global Step (# training iters)")
    plt.title(f"{name}: Loss")
    plt.legend(loc="upper right")

    fig.canvas.draw()
    fig.savefig(img_path, dpi=300)
    plt.close()

def save_alpha_grid(pred_alpha, fig_path, timestamps):
    """## Save a batch of grids of predicted occupancies as .npy files"""
    alpha_np = pred_alpha.clone().detach().cpu().numpy()
    B = alpha_np.shape[0]

    for b in range(B):
        fname = f"{str(timestamps[b]).split('.')[0]}.npy"
        np.save(fig_path / fname, alpha_np[b,:])

def arrays_to_gif(arrays, gif_filename, fps=60):
    """## Convert a list of numpy arrays to a GIF"""
    images = [np.flipud(img).astype('uint8') for img in arrays]
    imageio.mimsave(gif_filename, images, fps=fps)