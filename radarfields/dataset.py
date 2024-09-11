from dataclasses import dataclass
from pathlib import Path
import json

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from radarfields.sampler import get_azimuths, get_range_samples
from utils.data import read_fft_image, read_LUT
from utils.train import range_to_world

@dataclass
class RadarDataset:
    device: str
    split: str

    # data
    data_path: str
    preprocess_file: str
    radar_dir: str = "radar"
    preprocess_dir: str = "preprocess_results"

    # sampling
    num_rays_radar: int = 200
    num_fov_samples: int = 10
    min_range_bin: int = 1 # NOTE: inclusive
    max_range_bin: int = 1200 # NOTE: also inclusive
    num_range_samples: int = 1199
    integrate_rays: bool = True
    bs: int = 10

    # optional settings
    sample_all_ranges: bool = False
    train_thresholded: bool = True
    reg_occ: bool = True
    additive: bool = False
    preload: bool = False
    square_gain: bool = True

    # radar intrinsics
    num_range_bins: int = 7536
    bin_size_radar: float = 0.044
    num_azimuths_radar: int = 400
    opening_h: float = 1.8
    opening_v: float = 40.0

    def __post_init__(self):
        self.training = self.split == "train"
        self.intrinsics_radar = (self.opening_h, self.opening_v)
        self.range_bounds = (self.min_range_bin, self.max_range_bin)

        # Load preprocess JSON
        self.project_root = Path(__file__).parent.parent
        preprocess_path = self.project_root / self.preprocess_dir
        with open(preprocess_path / self.preprocess_file.strip("\""),) as f:
            print(f"Loading in data from: {self.preprocess_file}")
            preprocess = json.load(f)
        self.preprocess = preprocess

        # Sampler
        self.indices = preprocess[self.split + '_indices']
        self.sampler = SubsetRandomSampler(self.indices)

        # Offsets and Scalers to normalize XYZ model queries
        self.offsets = torch.tensor(preprocess["offsets"], device=self.device)
        self.scalers = torch.tensor(preprocess["scalers"], device=self.device)

        # Sensor poses
        self.poses_radar = []
        for f in tqdm.tqdm(preprocess["radar2worlds"], desc=f"Loading radar poses"):
            pose_radar = np.array(f, dtype=np.float32) # [4, 4]
            self.poses_radar.append(pose_radar)

        # FFT data
        fft_path = self.project_root / self.data_path / self.radar_dir
        fft_frames = preprocess["timestamps_radar"]
        self.timestamps = fft_frames
        self.fft_frames = []
        if self.train_thresholded: # train on thresholded FFT data
            thresh_path = self.project_root / 'preprocess_results' / 'thresholded_fft' / Path(self.data_path).name
            for fft_frame in tqdm.tqdm(fft_frames, desc=f"Loading (thresholded) FFT data"):
                thresholded_fft = torch.tensor(np.load(thresh_path / (str(fft_frame).split('.')[0] + '.npy')))
                self.fft_frames.append(thresholded_fft)
        else: # Train on unprocessed FFT data
            for fft_frame in tqdm.tqdm(fft_frames, desc=f"Loading FFT data"):
                raw_radar_fft = read_fft_image(fft_path / fft_frame)
                self.fft_frames.append(raw_radar_fft)

        # Occupancy components (for regularizing occupancy)
        if self.reg_occ:
            self.occ_frames = []
            occ_path = self.project_root / 'preprocess_results' / 'occupancy_component' / str(self.preprocess_file).split('.')[0]
            for fft_frame in tqdm.tqdm(fft_frames, desc=f"Loading occupancy components"):
                timestamp = str(fft_frame).split('.')[0] + '.npy'
                occupancy_component = torch.tensor(np.load(occ_path / timestamp), dtype=torch.float32)
                self.occ_frames.append(occupancy_component)
        
        if not self.training: # Sample all azimuth-range bins during testing
            self.num_rays_radar = self.num_azimuths_radar
            self.num_range_samples = self.max_range_bin-self.min_range_bin+1
            self.sample_all_ranges = True
        if (self.max_range_bin-self.min_range_bin+1) ==self.num_range_samples: self.sample_all_ranges = True
        
        # Radiation pattern look-up table (LUT) for integrating beam samples
        elevation_LUT_path = (self.project_root / self.data_path).parent / "elevation.csv"
        azimuth_LUT_path = (self.project_root / self.data_path).parent / "azimuth.csv"
        print(f"loading elevation radiation pattern from {elevation_LUT_path}")
        self.elevation_LUT_linear = read_LUT(elevation_LUT_path)
        print(f"loading azimuth radiation pattern from {azimuth_LUT_path}")
        self.azimuth_LUT_linear = read_LUT(azimuth_LUT_path)

        # Converting to torch
        self.poses_radar = torch.from_numpy(np.stack(self.poses_radar, axis=0)) # [B, 4, 4]
        self.fft_frames = torch.from_numpy(np.stack(self.fft_frames, axis=0)).float() # [B, H, W]
        if self.reg_occ: self.occ_frames = torch.stack(self.occ_frames, dim=0) # [B, H, W]

        # If enabled, preload all data onto GPU memory
        if self.preload:
            self.poses_radar = self.poses_radar.to(self.device)
            self.fft_frames = self.fft_frames.to(self.device)
            if self.reg_occ: self.occ_frames = self.occ_frames.to(self.device)

    def collate(self, index):
        '''
        ## Custom collate_fn to collate a batch of raw FFT data.
        ### Also samples range-azimuth bins for each FFT frame in the batch.
        '''
        B = len(index) # index is a list of length [B]
        N = self.num_rays_radar
        S = self.num_fov_samples
        R = self.num_range_samples

        results = {}

        # Radar2world matrices
        poses_radar = self.poses_radar[index].to(self.device)  # [B, 4, 4]

        # Sample azimuth angles at which to query model (in terms of idx from 0 -> 400)
        # (sample all azimuths during test)
        azimuth_samples = get_azimuths(B, N, self.num_azimuths_radar, self.device, all=not self.training)

        # Sample range bins at which to query model, and convert to radial distances in meters
        range_samples_idx = get_range_samples(B, self.num_rays_radar,
                                          self.num_range_samples,
                                          self.range_bounds,
                                          device=self.device,
                                          all=self.sample_all_ranges) # [B, N, R]
        range_samples = range_to_world(range_samples_idx, self.bin_size_radar) # [B, N, R]
        range_samples_expanded = range_samples.repeat_interleave(S, dim=1) # [B, N*S, R]

        # Crop radar FFT frames & occupancy components to provided bin ranges
        # NOTE: bins are 1-indexed, tensors are 0-indexed
        fft = self.fft_frames[index].to(self.device)
        fft = fft[:,:,self.min_range_bin-1:self.max_range_bin] # [B, H, W]
        if self.reg_occ:
            occ = self.occ_frames[index].to(self.device)
            occ = occ[:,:,self.min_range_bin-1:self.max_range_bin] # [B, H, W]

        results.update({
            "intrinsics": self.intrinsics_radar, # (fov_h, fov_v)
            "bs": B,
            "num_rays_radar": N,
            "num_fov_samples": S,
            "num_range_samples": R,
            "indices": index, # batch FFT frame indices
            "poses": poses_radar, # [B, 4, 4]
            "azimuths": azimuth_samples, # [B, N]
            "ranges": range_samples_expanded, # [B, N*S, R]
            "ranges_original": range_samples, # [B, N, R]
            "offsets": self.offsets,
            "scalers": self.scalers,
            "timestamps": [self.timestamps[i] for i in index], # batch FFT frame timestamps
            "azim_LUT": self.azimuth_LUT_linear,
            "elev_LUT": self.elevation_LUT_linear,
            "fft": fft # [B, H, R]
            })
        if self.reg_occ: results["occ"] = occ # [B, H, R]

        if not self.training: return results # We test on all FFT bins

        # If training, filter data for only sampled azimuth beams
        num_bins = self.max_range_bin-self.min_range_bin+1
        results["fft"] = torch.gather(fft, 1, azimuth_samples[...,None].expand((B,N,num_bins)))
        if self.reg_occ: results["occ"] = torch.gather(occ, 1, azimuth_samples[...,None].expand((B,N,num_bins)))

        # If sampling all range bins per-ray, no need to filter the data anymore; return
        if self.sample_all_ranges: return results

        # Else use only the sampled range bins
        results["fft"] = torch.gather(results["fft"], 2, range_samples_idx-self.min_range_bin)
        if self.reg_occ: results["occ"] = torch.gather(results["occ"], 2, range_samples_idx-self.min_range_bin)
        return results
    
    def dataloader(self, batch_size):
        size = len(self.poses_radar)
        loader = DataLoader(
            list(range(size)),
            batch_size=batch_size,
            collate_fn=self.collate,
            num_workers=0,
            sampler=self.sampler,
            pin_memory=False
        )
        loader._data = self
        loader.num_poses = size
        return loader