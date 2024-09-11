import torch
import numpy as np

from utils.train import range_to_world

def rcs_to_intensity(rcs, ranges, offset=0.05, scaler=1.0, approx=True):
    '''
    ## Compute FFT measurement amplitude from radar cross-section (RCS)
    NOTE: FFT data is in 2*Db, w/ noise floor applied & normalization
    '''
    if approx: return torch.log10(rcs + offset) * scaler
    return torch.log10(rcs / ranges**2 + offset) * scaler

def avg_rays(samples, num_samples):
    '''## Average a batch of super-sampled rays with S=num_samples super-samples per ray.'''
    if num_samples == 1: return torch.squeeze(samples, dim=-1) # No super-sampling; return
    
    B, NS, R, F = samples.shape
    S = num_samples

    samples_reshaped = samples.reshape((B, -1, S, R, F)) # [B, N, S, R, 1]
    ray_sums = torch.sum(samples_reshaped, dim=2) # [B, N, R, 1]
    ray_sums = ray_sums / S # [B, N, R, 1]
    return ray_sums.squeeze(dim=-1) # [B, N, R]

def get_weights(offsets, azim_LUT, elev_LUT, device):
    '''
    ## Query azimuth and elevation LUTs
    NOTE: offsets (in degrees) should be on CPU, and are shape [..., 3]
    '''
    azim_weights = np.interp(offsets[...,2].numpy(), azim_LUT[:,0], azim_LUT[:,1])
    azim_weights = torch.from_numpy(azim_weights).type(torch.float32).to(device)
    elev_weights = np.interp(offsets[...,1].numpy(), elev_LUT[:,0], elev_LUT[:,1])
    elev_weights = torch.from_numpy(elev_weights).type(torch.float32).to(device)
    return azim_weights*elev_weights # [B, N*S] or [B, N*S, R]

def integrate_rays_LUT(samples, angular_offsets, num_samples, azim_LUT, elev_LUT, device):
    '''
    ## Integrate a batch of sampled rays with S=num_samples super-samples per ray.
    ### This is a weighted integration, where each sample is weighted by angular offset\
    using azim_LUT and elev_LUT.
    (NOTE: this is used to integrate batches of queries at train & test time)
    '''
    if num_samples == 1: return torch.squeeze(samples, dim=-1) # No super-sampling; return 
    
    B, NS, R, F = samples.shape
    S = num_samples
    
    # Get integration weights
    all_weights = get_weights(angular_offsets, azim_LUT, elev_LUT, device) # [B, N*S] or [B, N*S, R]

    # Compute total weight per super-sampled bundle
    if angular_offsets.dim == 4: all_weights = all_weights.reshape((B, -1, S, R))[..., None] # [B, N, S, R, 1]
    else: all_weights = all_weights.reshape((B, -1, S)) [..., None, None] # [B, N, S, 1, 1]
    all_weight_sums = torch.sum(all_weights, dim=2) # [B, N, R, 1] or [B, N, 1, 1]

    # Weighted average
    samples_reshaped = samples.reshape((B, -1, S, R, F)) # [B, N, S, R, 1]
    ray_sums = torch.sum(samples_reshaped*all_weights, dim=2) # [B, N, R, 1]
    integrated_rays = ray_sums / all_weight_sums # [B, N, R, 1]
    return integrated_rays.squeeze(dim=-1) # [B, N, R]

def integrate_points_LUT(samples, angular_offsets, num_samples, azim_LUT, elev_LUT, device):
    '''
    ## Integrate a set of P points with S=num_samples super-samples per point.
    ### This is a weighted integration, where each sample is weighted by angular offset\
    using azim_LUT and elev_LUT.
    (NOTE: this function is used to integrate arbitrary sets of points)
    '''
    if num_samples == 1: return torch.squeeze(samples, dim=-1) # No super-sampling; return 
    
    PS, F = samples.shape
    S = num_samples
    
    # Get integration weights
    all_weights = get_weights(angular_offsets, azim_LUT, elev_LUT, device) # [P*S]

    # Compute total weight per super-sampled bundle
    all_weights = all_weights.reshape((-1, S))[..., None] # [P, S, 1]
    all_weight_sums = torch.sum(all_weights, dim=1) # [P, 1]

    samples_reshaped = samples.reshape((-1, S, F)) # [P, S, 1]
    point_sums = torch.sum(samples_reshaped*all_weights, dim=1) # [P, 1]
    integrated_points = point_sums / all_weight_sums # [P, 1]
    return integrated_points.squeeze(dim=-1) # [P]

def compute_occupancy_component(fft_img, occupancy_threshold, highlights_threshold, min_range_bin, binarize=True):
    '''
    ## Thresholds all range-azimuth bins with intensity above occupancy_threshold,\
    and clips all intensities to a max. value of highlights_threshold.
    ## If binarize=True, casts all non-zero bins to 1.0

    NOTE: first min_range_bin range bins are ignored, since these catch the vehicle's
    roof
    '''
    fft = fft_img.clone()
    fft[:,:min_range_bin] = 0.0
    occupied = fft > occupancy_threshold
    occupancy_component = fft * occupied
    occupancy_component = torch.clip(occupancy_component, max=highlights_threshold)

    # Converting to binary mask
    if binarize:
        has_signal = occupancy_component > 0.01
        occupancy_component[has_signal] = 1.0

    return occupancy_component

def update_probability(current_prob, evidence_prob):
    """Bayesian update"""
    return (evidence_prob * current_prob) / ((evidence_prob * current_prob) + (1 - evidence_prob) * (1 - current_prob))

def bayesian_polar_occupancy_map(thresholded_fft, bin_ranges, i_offset=-0.15, i_exp=2.0, decay_bins=10):
    """
    ## Compute a radially-integrated occupancy probability map using a bayesian updated rule

    intensity -> occupancy probability mapping:
    prob. of occupancy = FFT_intensity * e^(i_exp * (FFT_intensity - i_offset))

    integrated probability decay:
    decay factor = e^(-1 * (range_bin_number * num_bins_since_previous_peak) / decay_bins)

    :param thresholded_fft (torch.Tensor): thresholded FFT measurement
    :param bin_ranges (tuple): [min_bin, max_bin], inclusive & 1-indexed
    :param i_offset (float): additive offset to intensity -> occupancy probability mapping
    :param i_exp (float): multiplicative scaler to intensity -> occupancy probability mapping
    :param decay_bins (float/int): decay rate of accumulated radial occupancy probability (in bins)
    :return occ_map (torch.Tensor): occupancy map
    :return integral_map (torch.Tensor): integrated occupancy map
    """
    num_azims = thresholded_fft.shape[0]
    min_bin, max_bin = bin_ranges
    occ_map = torch.full_like(thresholded_fft, 0.5)
    integral_map = torch.full_like(thresholded_fft, 0.5)
    integral_probs = torch.zeros((num_azims)) # [400]
    idx_object_start = (min_bin-1)*torch.ones((num_azims)) # [400]

    for bin_num in torch.arange(min_bin-1, max_bin):
        intensities = thresholded_fft[:, bin_num] # [400]
        exp_decay = np.exp(-(bin_num - idx_object_start) / decay_bins) # decay factor
        integral_probs = exp_decay*integral_probs
        intensity_probs = np.clip(intensities*np.exp(i_exp*(intensities-i_offset)), 0.0, 0.999) # [400]
        idx = integral_probs<intensity_probs
        idx_object_start[idx] = bin_num # Tracking num. bins since previous relative max.
        integral_probs = np.maximum(integral_probs, intensity_probs) # [400]       

        occ_map[:, bin_num] = update_probability(occ_map[:, bin_num], integral_probs)
        integral_map[:, bin_num] = integral_probs

    return occ_map, integral_map

def compute_spherical_grid_noise_threshold(fft_img, min_range, max_range):
    '''
    ## Dynamically threshold all range-azimuth bins with a grid\
    of range-wise and azimuth-wise medians
    '''
    fft = fft_img.clone()

    # Compute per-bin threshold grid
    medians_range, _ = torch.median(fft[:, min_range:max_range+1], axis=1) # [400]
    medians_azimuth, _ = torch.median(fft,axis=0)
    grid = torch.maximum(medians_azimuth.unsqueeze(0), medians_range.unsqueeze(1))
    has_signal = fft > 1.5*grid # [400, 7536]

    return fft * has_signal

def adaptive_gaussian_kernel(ranges, base_sigma=0.044, scaling_factor=0.01, kernel_size=21, bin_size=0.044):
    """
    ## Generate a Gaussian kernel array where sigma increases with radius

    :param ranges (tensor): tensor of ranges, in meters
    :param base_sigma (float): Base value of sigma
    :param scaling_factor (float): Factor by which sigma increases with radius
    :param kernel_size (int): Must be odd
    :param bin_size (float): radial length of bin (meters)
    :return ndarray: Gaussian kernels with radially-varying sigma
    """
    if kernel_size % 2 == 0: kernel_size = kernel_size + 1

    # Radial offsets relative to center of each kernel
    offset = torch.arange(-kernel_size//2+1,kernel_size//2+1)*bin_size # [K=kernel_size]
    sigma = base_sigma + scaling_factor * ranges # [R]

    # Return normalized gaussians
    gaussian = torch.exp(-0.5 * ((offset.unsqueeze(0))/ sigma.unsqueeze(1)) ** 2) # [R, K]
    normalization = torch.sum(gaussian,axis=1).unsqueeze(1) # [R, 1]
    return gaussian / normalization # [R, K]

def adaptive_gaussian_kernel_phi(sigma=2, kernel_size=21):
    """
    ## Generate a Gaussian kernel array for blurring along azimuth

    NOTE: sigma is not adaptive to radial distance, as azimuthal cross-talk
    is fixed and therefore range-invariant

    :param sigma (float): Base value of sigma
    :param kernel_size (int): Must be odd
    :return ndarray: Gaussian kernel with varying sigma
    """
    # Azimuthal offsets relative to center of each kernel
    offset = torch.arange(-kernel_size//2+1,kernel_size//2+1)

    gaussian = torch.exp(-0.5 * ((offset)/ torch.tensor(sigma)) ** 2)
    normalization = torch.sum(gaussian)
    return gaussian / normalization

def apply_radial_deblur_richardson(image, base_sigma=0.088, base_sigma_phi=0.8, scaling_factor=0.044, iterations=5,
                                   filter_epsilon=0.001, max_range_bin=7536, bin_size=0.044, clip=True):
    """
    ## Apply an adaptive radial deblur to an image in polar coordinates

    :param image (ndarray): Image in polar coordinates (radial distance, angle)
    :param base_sigma (float): Base sigma for radial Gaussian kernel
    :param base_sigma_phi (float): Base sigma for azimuthal Gaussian kernel
    :param scaling_factor (float): Scaling factor for sigma based on radius
    :param iterations (int): number of deblurring iterations
    :param filter_epsislon (float): threshold for blurred img to prevent noise amplification
    :param max_range_bin (int): number of range bins in radar
    :param bin_size (float): range bin size (in meters)
    :return ndarray: Deblurred image
    """
    im_deconv = torch.full(image.shape, 0.5)
    unfolded_deconv = torch.Tensor.unfold(torch.nn.functional.pad(im_deconv,(10,10), mode="reflect"), 1, 21, 1)
    radius = range_to_world(torch.arange(max_range_bin)+1, bin_size)  # all bin ranges, in meters
    kernel = adaptive_gaussian_kernel(radius, base_sigma, scaling_factor) # range blur kernel
    kernel_phi = adaptive_gaussian_kernel_phi(sigma=base_sigma_phi).type(torch.float32) # azimuth blur kernel

    for _ in range(iterations):

        conv = torch.sum(unfolded_deconv*kernel.unsqueeze(0), axis=-1)
        conv = torch.nn.functional.conv1d(torch.transpose(conv,0,1).unsqueeze(1),kernel_phi.unsqueeze(0).unsqueeze(0),padding=10).squeeze().transpose(0,1)
        
        if filter_epsilon: relative_blur = torch.where(conv < filter_epsilon, 0, image / conv)
        else: relative_blur = image / conv
        
        unfolded_relative_blur = torch.Tensor.unfold(torch.nn.functional.pad(relative_blur,(10,10), mode="reflect"), 1, 21, 1)
        im_deconv *= torch.sum(unfolded_relative_blur*kernel.unsqueeze(0), axis=-1)
        unfolded_deconv = torch.Tensor.unfold(torch.nn.functional.pad(im_deconv,(10,10), mode="reflect"), 1, 21, 1)
    
    if clip: im_deconv = torch.clip(im_deconv, 0.0, 1.0)
    return im_deconv