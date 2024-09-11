from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import glob
import tqdm

from radarfields.sampler import get_points
from radarfields.radar import avg_rays, integrate_rays_LUT, rcs_to_intensity, integrate_points_LUT
from radarfields.nn.pose_refinement import PoseOptimizer
from utils.vis import save_loss_plot, save_alpha_grid, plot_trajectory, render_FFT_batch
from radarfields.figures import render_outputs_over_depth, render_outputs_supersampled

class Trainer(object):
    def __init__(self, args, model, split, criterion, optimizer, lr_scheduler,
                 device=None, max_keep_ckpt=2, scheduler_update_every_step=True, skip_ckpt=False):
        self.args = args
        self.name = str(args.name).strip("\"") # checkpoint name
        self.training = split == "train"
        self.workspace = Path(args.workspace) # workspace directory to save logs & ckpts
        self.max_keep_ckpt = max_keep_ckpt # max num of saved ckpts in disk
        self.which_checkpoint = args.ckpt # which checkpoint to use at init time
        self.refine_poses = self.args.refine_poses
        self.skip_ckpt = skip_ckpt # True if loading demo checkpoint later
        self.learned_norm = self.args.learned_norm
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = (device if device else torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu"))
        self.log_ptr = None

        model.to(self.device)
        self.model = model
        self.criterion = criterion

        if optimizer is None: # dummy optimizer
            self.optimizer = torch.optim.Adam(
                self.model.get_params(args.lr), betas=(0.9, 0.99), eps=1e-15
            )
        else: self.optimizer = optimizer(self.model)

        if self.learned_norm:
            self.offset = torch.nn.Parameter(torch.tensor([self.args.initial_offset]).to(self.device), requires_grad=True)
            self.scaler = torch.nn.Parameter(torch.tensor([self.args.initial_scaler]).to(self.device), requires_grad=True)
            self.optimizer.add_param_group({'params': [self.offset], 'lr': self.args.lr_offset})
            self.optimizer.add_param_group({'params': [self.scaler], 'lr': self.args.lr_scaler})
        else:
            self.offset = self.args.initial_offset
            self.scaler = self.args.initial_scaler

        if lr_scheduler is None: # dummy lr scheduler
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1
            )
        else: self.lr_scheduler = lr_scheduler(self.optimizer)

        # Learned pose refinement
        if self.refine_poses and not self.skip_ckpt: # skip if loading demo
            self.pose_model = PoseOptimizer(self.args.all_poses, self.args.pose_mode, self.args.colinear, self.device,
                                            non_trainable_camera_indices=self.args.test_indices if self.training else None)

            self.pose_optimizer = torch.optim.Adam(
                self.pose_model.get_params(args.pose_lr), betas=(0.9, 0.99), eps=1e-15
            )

            # Lambda LR scheduler
            # (decay to 0.1 * init_lr at last iter step)
            if self.args.schedule_pose:
                self.pose_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.pose_optimizer, lambda iter: 0.1 ** min(iter / self.args.iters, 1)
                )
            else: self.pose_lr_scheduler = None
        else:
            self.pose_model = None
            self.pose_optimizer = None
            self.pose_lr_scheduler = None

        self.softmax = torch.nn.Softmax(dim=1)

        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.loss_dict = {
            "total loss": [],
            "pose gradient penalty": [],
            "pose coplanar penalty": [],
            "occ. bimodal penalty": [],
            "occ. mask penalty": [],
            "FFT reconstruction": [],
            "occupancy grounding penalty": [],
            "occupancy above penalty": []
        }
        self.checkpoints = [] # paths to saved checkpoints

        # Coarse-to-fine masking
        if self.args.mask: self.sin_epoch = 0.0 if self.training else 1.0
        else: self.sin_epoch = None

        # Prepare workspace & logger
        self.workspace.mkdir(exist_ok=True)
        self.log_path = self.workspace / "logs"
        self.log_path.mkdir(exist_ok=True)
        self.log_path = self.log_path / f"log_{self.name}.txt"
        self.log_ptr = open(self.log_path, "a+")
        self.ckpt_path = Path("checkpoints")
        self.ckpt_path.mkdir(exist_ok=True)
        self.img_path = self.workspace / "imgs" / self.name
        self.img_path.mkdir(parents=True, exist_ok=True)
        self.alpha_grid_path = self.img_path / "alpha_results"
        self.alpha_grid_path.mkdir(exist_ok=True)
        self.FFT_path = self.img_path / "FFT"
        self.FFT_path.mkdir(exist_ok=True)
        self.plot_path = self.workspace / "plots" / self.name
        self.plot_path.mkdir(parents=True, exist_ok=True)
        self.trajectories_path = self.img_path / "trajectories"
        self.trajectories_path.mkdir(exist_ok=True)

        self.log(f'[INFO] Trainer: {self.name} | {self.device} | workspace: {self.workspace}')
        self.log(f"[INFO] # of parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")
        self.log(f"[INFO] Writing outputs to workspace directory: {self.workspace}")

        # Checkpoints
        if skip_ckpt: return
        if self.which_checkpoint == "scratch": self.log("[INFO] Training from scratch ...")
        elif self.which_checkpoint == "latest":
            self.log("[INFO] Loading latest checkpoint ...")
            self.load_checkpoint()
        elif self.which_checkpoint == "latest_model":
            self.log("[INFO] Loading latest checkpoint (model only)...")
            self.load_checkpoint(model_only=True)
        else:  # path to specific checkpoint
            self.log(f"[INFO] Loading {self.which_checkpoint} ...")
            self.load_checkpoint(self.which_checkpoint)
    
    def __del__(self):
        if self.log_ptr: self.log_ptr.close()

    def log(self, *args, **kwargs):
        '''## Log message and print it to Stdout'''
        print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()  # write immediately to file

    def save_checkpoint(self, name=None, full=False, remove_old=True, loader=None):
        '''## Save model checkpoint as .pth file'''
        if name is None: name = f"{self.name}_ep{self.epoch:04d}"
        if loader is not None: name = f"{self.name}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss_dict": self.loss_dict,
            "checkpoints": self.checkpoints,
        }

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            if self.refine_poses:
                state["pose_optimizer"] = self.pose_optimizer.state_dict()
                if self.pose_lr_scheduler:
                    state["pose_lr_scheduler"] = self.pose_lr_scheduler.state_dict()

        if self.learned_norm:
            state["offset"] = self.offset.clone().detach()
            state["scaler"] = self.scaler.clone().detach()

        state["model"] = self.model.state_dict()
        if self.refine_poses: state["pose_model"] = self.pose_model.state_dict()

        file_path = f"{self.ckpt_path}/{name}.pth"

        if remove_old:
            self.checkpoints.append(file_path)
            if len(self.checkpoints) > self.max_keep_ckpt:
                old_ckpt = Path(self.checkpoints.pop(0))
                if old_ckpt.exists(): old_ckpt.unlink()

        if loader: state["loader"] = loader

        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None, model_only=False, demo=False):
        '''## Load in .pth model checkpoint file'''
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                if self.args.demo_name is None: self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict["model"], strict=False
        )
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if demo:
            loader = checkpoint_dict["loader"]
            self.args.all_poses = loader._data.poses_radar.to(self.args.device)
            self.args.test_indices = loader._data.preprocess["test_indices"]

            if self.refine_poses:
                self.pose_model = PoseOptimizer(self.args.all_poses, self.args.pose_mode, self.args.colinear, self.device,
                                                non_trainable_camera_indices=self.args.test_indices if self.training else None)

        if self.refine_poses:
            if "pose_model" not in checkpoint_dict:
                raise RuntimeError("ERROR: --refine_poses=True, but no pose model in checkpoint.")

            missing_keys, unexpected_keys = self.pose_model.load_state_dict(
                checkpoint_dict["pose_model"], strict=False
            )
            self.log("[INFO] loaded pose model.")
            if len(missing_keys) > 0:
                self.log(f"[WARN] missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if model_only: return

        self.loss_dict = checkpoint_dict["loss_dict"]
        self.checkpoints = checkpoint_dict["checkpoints"]
        self.epoch = checkpoint_dict["epoch"]
        self.global_step = checkpoint_dict["global_step"]
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except: self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except: self.log("[WARN] Failed to load scheduler.")
        
        if self.refine_poses:
            if demo:
                self.pose_optimizer = torch.optim.Adam(
                    self.pose_model.get_params(self.args.pose_lr), betas=(0.9, 0.99), eps=1e-15
                )

                # Lambda LR scheduler
                # (decay to 0.1 * init_lr at last iter step)
                if self.args.schedule_pose:
                    self.pose_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        self.pose_optimizer, lambda iter: 0.1 ** min(iter / self.args.iters, 1)
                    )
                else: self.pose_lr_scheduler = None

            if self.pose_optimizer and "pose_optimizer" in checkpoint_dict:
                try:
                    self.pose_optimizer.load_state_dict(checkpoint_dict["pose_optimizer"])
                    self.log("[INFO] loaded pose optimizer.")
                except: self.log("[WARN] Failed to load pose optimizer.")

            if self.pose_lr_scheduler and "pose_lr_scheduler" in checkpoint_dict:
                try:
                    self.pose_lr_scheduler.load_state_dict(checkpoint_dict["pose_lr_scheduler"])
                    self.log("[INFO] loaded pose scheduler.")
                except: self.log("[WARN] Failed to load pose scheduler.")

        if self.learned_norm:
            self.offset.data.copy_(checkpoint_dict["offset"].to(self.device))
            self.scaler.data.copy_(checkpoint_dict["scaler"].to(self.device))
        
        if demo: return loader

    def save_figures(self, model_outputs, data_batch, points, loader):
        timestamps = data_batch["timestamps"]
        
        # Save occupancy predictions as .npy files
        save_alpha_grid(model_outputs[3], self.alpha_grid_path, timestamps)

        # render FFT measurements
        if self.args.render_fft:
            render_FFT_batch(data_batch["fft"], self.args.max_test_bin,
                             self.args.intrinsics_radar["bin_size_radar"],
                             self.args.intrinsics_radar["num_azimuths_radar"],
                             self.FFT_path, timestamps, self.args.min_range_bin,
                             resolution=self.args.figure_resolution,
                             noise_floor=self.args.noise_floor)
        
        bounds = loader._data.range_bounds
        bounds = (bounds[0], self.args.max_test_bin)
        render_outputs_supersampled(self, data_batch, points, bounds, self.img_path, timestamps,
                                    resolution=self.args.figure_resolution)
        render_outputs_over_depth(self, data_batch, points, bounds, self.img_path, timestamps,
                                    resolution=self.args.figure_resolution)

    def train(self, train_loader, max_epochs):
        '''## Run model training procedure'''

        # If refining poses, plot the original poses
        if self.refine_poses: plot_trajectory(train_loader._data.poses_radar[1:,...], self.trajectories_path / "original_trajectory.png")

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            # Used for coarse-to-fine masking
            if self.args.mask: self.sin_epoch = min(1.0, 0.05 + np.sin(self.epoch/(max_epochs - 1) * np.pi/2))

            self.train_epoch(train_loader)
            self.save_checkpoint(full=True)

        if self.args.save_loss_plot: save_loss_plot(self.loss_dict, self.plot_path, self.args, self.global_step)
        if self.refine_poses: plot_trajectory(self.pose_model.get_corrected_poses(), self.trajectories_path / "refined_train_trajectory.png")
    
    def test(self, loader):
        '''## Run testing procedure'''
        # Clearing cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if self.refine_poses: # Interpolate between refined train poses to get corrected test poses
            self.pose_model.interp_test_poses(self.args.test_indices)
            plot_trajectory(self.pose_model.get_corrected_poses(), self.trajectories_path / "refined_all_trajectory")

        self.test_epoch(loader)

        if self.args.demo: self.save_checkpoint(full=True, loader=loader, remove_old=False)
    
    def train_epoch(self, loader):
        '''## Single training epoch'''
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = torch.tensor([0.0], device=self.device)
        self.local_step = 0

        self.model.train()
        if self.refine_poses: self.pose_model.train()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for data in loader:
            self.local_step += 1
            self.global_step += 1
            
            self.optimizer.zero_grad()
            if self.refine_poses: self.pose_optimizer.zero_grad()

            if self.refine_poses: data["poses"] = self.pose_model.apply_to_poses(data["indices"])

            points = get_points(data, self.args, self.device) # model queries
            out = self.predict_waveform(data, points)
            loss = self.compute_loss(data, points, *out)
            loss.backward()

            self.optimizer.step()
            if self.refine_poses: self.pose_optimizer.step()

            if self.scheduler_update_every_step:
                if self.lr_scheduler: self.lr_scheduler.step()
                if self.pose_lr_scheduler: self.pose_lr_scheduler.step()

            total_loss += loss
            pbar.set_description(
                f"loss={loss} ({(total_loss/self.local_step).item()})"
            )
            pbar.update(loader.batch_size)
        pbar.close()

        if not self.scheduler_update_every_step:
            if self.lr_scheduler: self.lr_scheduler.step()
            if self.pose_lr_scheduler: self.pose_lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def test_epoch(self, loader):
        '''## Single test epoch'''
        self.log(f"++> Test at epoch {self.epoch} ...")

        total_loss = 0
        
        self.model.eval()
        if self.refine_poses: self.pose_model.eval()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                if self.refine_poses: data["poses"] = self.pose_model.apply_to_poses(data["indices"])

                points = get_points(data, self.args, self.device) # model queries
                out = self.predict_waveform(data, points)
                loss = self.compute_loss(data, points, *out)

                loss_val = loss.item()
                total_loss += loss_val

                # Visualize model predictions per-frame
                self.save_figures(out, data, points, loader)

                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        pbar.close()

        self.log(f"++> Test epoch {self.epoch} Finished.")

    def compute_loss(self, data, points, pred_fft, alpha_integrated, rd_integrated, alpha):
        """
        ## Compute loss terms
        ### (including regularization penalties)
        """
        B = data["bs"]
        N = data["num_rays_radar"]
        S = data["num_fov_samples"]
        R = data["num_range_samples"]

        loss_dict = dict()
        
        fft = data["fft"]  # [B, N, R]
        alpha_integrated_fp32 = alpha_integrated.type(torch.float32) # [B, N, R, 1]
        alpha_fp32 = alpha.type(torch.float32) # [B, N*S, R]

        # Primary FFT reconstruction loss
        loss = self.criterion["fft"](pred_fft, fft) * self.args.weight_fft
        loss_dict["FFT reconstruction"] = loss

        # Occupancy component regularization/supervision
        if self.args.reg_occ:
            occ = data["occ"]
            if self.args.occ_loss == "kl":
                pred_occ_distribution = alpha_integrated_fp32 / torch.sum(alpha_integrated_fp32)
                occ_distribution = occ / torch.sum(occ)
                occ_penalty = self.criterion["occ"](torch.squeeze(pred_occ_distribution, dim=-1).log(), occ_distribution)
            else: occ_penalty = self.criterion["occ"](torch.squeeze(alpha_integrated, dim=-1), occ)
            occ_penalty = occ_penalty * self.args.weight_occ
            loss = loss + torch.nan_to_num(occ_penalty)
            loss_dict["occ. mask penalty"] = occ_penalty

            if self.args.bimodal:
                mask_bim = occ > 0.01
                bimodal_penalty = torch.std(alpha_integrated_fp32[mask_bim]) + torch.std(alpha_integrated_fp32[~mask_bim])
                bimodal_penalty = bimodal_penalty * self.args.weight_bimodal
                loss = loss + torch.nan_to_num(bimodal_penalty)
                loss_dict["occ. bimodal penalty"] = bimodal_penalty

            alpha_fp32 = alpha_fp32.reshape((B, N, S, R))
            angular_offsets = points["angular_offsets"].to(self.device) # [B, N*S, 3] or [B, N*S, R, 3]
            last_dims = angular_offsets.shape[2:] # [3] or [R, 3]
            angular_offsets = angular_offsets.reshape(B, N, S, *last_dims) # [B, N, S, 3] or [B, N, S, R, 3]
            if len(last_dims) == 2: angular_offsets = angular_offsets[:, :, :, :, 1] # [B, N, S, R]
            else: angular_offsets = angular_offsets[:, :, :, 1][..., None].expand((B, N, S, R))

            # Occupancy grounding penalty
            if self.args.ground_occ:               
                grads = torch.zeros((S-1), device=self.device)
                for s in range(1, S):
                    running_total = torch.zeros((B,N,R), device=self.device)
                    for i in range(0, s):
                        dz = angular_offsets[:,:,s,:] - angular_offsets[:,:,i,:] + 1e-4
                        da = alpha_fp32[:,:,s,:] - alpha_fp32[:,:,i,:] + 1e-4
                        running_total = running_total + ((da/dz) + 1e-4) * (da < 0.0)
                    grads[s-1] = torch.mean(running_total)
                grounding_penalty = torch.mean(grads) * -1.0 * self.args.weight_ground_occ
                loss = loss + torch.nan_to_num(grounding_penalty)
                loss_dict["occupancy grounding penalty"] = grounding_penalty

            # Above mounting height penalty
            if self.args.penalize_above:           
                # Only penalizing occupancy deposited above radar mounting height
                above_sensor = angular_offsets < 0.0
                above_penalty = alpha_fp32 * (angular_offsets * -1.0) * above_sensor
                above_penalty = torch.mean(above_penalty) * self.args.weight_above
                loss = loss + torch.nan_to_num(above_penalty)
                loss_dict["occupancy above penalty"] = above_penalty

        # Pose refinement regularization
        if self.refine_poses:       
            if self.args.reg_poses:
                pose_penalty = self.pose_model.get_regularization_penalty() * self.args.weight_pose_reg
                loss = loss + pose_penalty
                loss_dict["pose gradient penalty"] = pose_penalty
            if self.args.reg_poses_coplanar:
                coplanar_penalty = self.pose_model.get_coplanarity_penalty() * self.args.weight_coplanar
                loss = loss + coplanar_penalty
                loss_dict["pose coplanar penalty"] = coplanar_penalty

        # If plotting loss terms, log them per-step
        if self.args.save_loss_plot:
            self.loss_dict["total loss"].append(loss.item())
            for loss_term in loss_dict.keys():
                self.loss_dict[loss_term].append(loss_dict[loss_term].item())
    
        return loss

    def predict_waveform(self, data, points, batch=True):
        '''
        ## Query Radar Fields w/ 3D points:
        ### pass outputs through forward model to reconstruct FFT measurement
        '''
        S = data["num_fov_samples"]
        xyz = points["xyz"] # [B, N*S, R, 3] cones of 3D points
        directions = points["directions"]  # [B, N*S, R, 3] view directions
        ranges = data["ranges"] # [B, N*S, R], where R is the # of sampled range bins
        azim = data["azim_LUT"] # Azimuth radiation profile look-up table
        elev = data["elev_LUT"] # Elevation radiation profile look-up table

        # Rescale queries to unit box
        offsets = data["offsets"].view(*([1] * (xyz.dim()-1) + [3]))
        scalers = data["scalers"].view(*([1] * (xyz.dim()-1) + [3]))
        xyz = xyz + offsets
        xyz = xyz / scalers

        # Query Radar Field
        out = self.model(xyz, directions, sin_epoch=self.sin_epoch)
        alpha = out["alpha"] # [B, N*S, R, 1]
        rd = out["rd"] # [B, N*S, R, 1]
        rcs = alpha*rd # radar cross-section (RCS)

        # Integrating 3D cone of NN samples ([B, N*S, R]) into a single,
        # 2D predicted azimuth measurement ([B, N, R])), using the radar antenna's
        # directional gain profile, stored as a LUT
        # (NOTE: we also store copies of alpha and r*d integrated individually,
        # as we use these for regularization and figure generation. Only the
        # integrated RCS is used to reconstruct the FFT measurement.)
        if self.args.integrate_rays:
            integrate = integrate_rays_LUT if batch else integrate_points_LUT
            alpha_integrated = integrate(alpha, points["angular_offsets"], S, azim, elev, self.device)
            rd_integrated = integrate(rd, points["angular_offsets"], S, azim, elev, self.device)
            rcs = integrate(rcs, points["angular_offsets"], S, azim, elev, self.device)
        else: # naive average
            alpha_integrated = avg_rays(alpha, S)
            rd_integrated = avg_rays(rd, S)
            rcs = avg_rays(rcs, S)

        # Computing FFT intensity
        pred_fft = rcs_to_intensity(rcs, ranges, offset=self.offset, scaler=self.scaler, approx=self.args.approximate_fft)
        
        return (pred_fft, alpha_integrated, rd_integrated, alpha[...,0])
