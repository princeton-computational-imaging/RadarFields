import configargparse
import yaml
from pathlib import Path

def get_arg_parser():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.ConfigparserConfigFileParser
        )

    # Global
    parser.add_argument("--config", is_config_file=True, help="config file path",)
    parser.add_argument("--name", type=str, default="radarfields", help="name of this training run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace", type=str, default="workspace")

    # Useful Toggles
    parser.add_argument("--test", action="store_true", help="toggle test mode")
    parser.add_argument("--skip_test", action="store_true", help="skip test during training")
    parser.add_argument("--voxels", action="store_true", help="toggle voxel grid visualization")
    parser.add_argument("--voxel_only", action="store_true", help="if True, do not render BEV outputs\
                        at test time; only render voxel grids")

    # Radar Intrinsics
    intrinsics_radar = {
        "opening_h": 1.8,
        "opening_v": 40.0,
        "num_azimuths_radar": 400,
        "num_range_bins": 7536,
        "bin_size_radar": 0.044,
        #"mounting_height": 1.62
    }
    parser.add_argument("--intrinsics_radar", type=yaml.safe_load, default=intrinsics_radar)

    # NN Backbone
    model_settings = {
        "in_dim": 3,
        "xyz_encoding": "HashGrid",
        "num_layers": 4,
        "hidden_dim": 64,
        "xyz_feat_dim": 32,
        "alpha_dim": 1,
        "alpha_activation": "sigmoid",
        "sigmoid_tightness": 1.0,
        "rd_dim": 1,
        "softplus_rd": True,
        "angle_dim": 3,
        "angle_in_layer": 3,
        "angle_encoding": "SphericalHarmonics",
        "num_bands_xyz": 10,
        "resolution": 512,
        "n_levels": 16,
        "bound": 1,
        "bn": True,
    }
    parser.add_argument("--model_settings", type=yaml.safe_load, default=model_settings)
    parser.add_argument("--tcnn", action="store_true", help="enable tcnn fully-fused NNs")

    # Training Options
    parser.add_argument("--iters", type=int, default=800, help="training iters")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--bs", type=int, default=10, help="batch size")
    parser.add_argument("--ckpt", type=str, default="latest", choices=["latest", "best"])
    parser.add_argument("--num_rays_radar", type=int, default=100,
                        help="# of azimuth angles to sample per radar frame")
    parser.add_argument("--num_fov_samples", type=int, default=10,
                        help="# of rays to sample within the opening angle of \
                        a radar, per azimuth sample")
    parser.add_argument("--num_range_samples", type=int, default=1125, help="# of range bins to sample")
    parser.add_argument("--sample_all_ranges", action="store_true", help="toggle sampling of all range bins per azimuth angle")
    parser.add_argument("--min_range_bin", type=int, default=76, help="lower bound on range bin sampling, inclusive")
    parser.add_argument("--max_range_bin", type=int, default=1200, help="upper bound on range bin sampling, inclusive")
    parser.add_argument("--mask", action="store_true", help="toggle coarse-to-fine masking of HashGrid features during training")
    parser.add_argument("--integrate_rays", action="store_true", help="toggle weighted average of ray super-samples using gain profile")
    parser.add_argument("--approximate_fft", action="store_true", help="toggle approximate FFT computation, instead of exact")
    parser.add_argument("--train_thresholded", action="store_true", help="toggle training on thresholded FFT scans")
    parser.add_argument("--learned_norm", action="store_true", help="toggle learned normalization on NN outputs")
    parser.add_argument("--initial_offset", type=float, default=1.0, help="initial offset applied when learning normalization")
    parser.add_argument("--initial_scaler", type=float, default=1.0, help="initial multiplicative scaler when learning normalization")
    parser.add_argument("--lr_offset", type=float, default=0.0, help="learning rate for learned offset")
    parser.add_argument("--lr_scaler", type=float, default=1e-3, help="learning rate for learned scaler")
    
    # Losses
    parser.add_argument("--fft_loss", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument("--weight_fft", type=float, default=0.60, help="weight to give FFT radar loss term")
    parser.add_argument("--reg_occ", action="store_true", help="toggle occupancy supervision with computed mask")
    parser.add_argument("--occ_loss", type=str, default="kl", choices=["kl", "mse"])
    parser.add_argument("--weight_occ", type=float, default=0.36)
    parser.add_argument("--bimodal", action="store_true", help="toggle bimodal occupancy reg. penalty")
    parser.add_argument("--weight_bimodal", type=float, default=0.03)
    parser.add_argument("--ground_occ", action="store_true", help="toggle occupancy grounding loss penalty")
    parser.add_argument("--weight_ground_occ", type=float, default=0.3)
    parser.add_argument("--penalize_above", action="store_true", help="penalize any occupancy deposited above sensor mounting height")
    parser.add_argument("--weight_above", type=float, default=0.05)

    # Dataset Options
    parser.add_argument("--seq", type=str, default="seq10", help="training seq. name")
    parser.add_argument("--radar_dir", type=str, default="radar")
    parser.add_argument("--preprocess_dir", type=str, default="preprocess_results")
    parser.add_argument("--preprocess_file", type=str, default="preprocess_results.json")
    parser.add_argument("--noise_floor", type=float, default=0.1525)
    parser.add_argument("--preload", action="store_true", help="toggle preload of data on GPU")
    parser.add_argument("--square_gain", action="store_true", help="square gain profile values")

    # Visualization/Figure Settings
    parser.add_argument("--render_fft", action="store_true", help="toggle rendering FFT data")
    parser.add_argument("--figure_resolution", type=int, default=781, help="resolution for figures")
    parser.add_argument("--save_loss_plot", action="store_true", help="periodically save loss plots")
    parser.add_argument("--max_test_bin", type=int, default=800, help="max bin at which to render outputs")

    # Learned Pose Refinement
    parser.add_argument("--refine_poses", action="store_true", help="toggle learned pose refinement")
    parser.add_argument("--pose_mode", type=str, default="SE3", choices=["SE3", "SO3xR3"],
                        help="which format to store pose refinements in")
    parser.add_argument("--reg_poses", action="store_true", help="toggle L2 penalty for poses")
    parser.add_argument("--reg_poses_coplanar", action="store_true", help="toggle coplanar regularization penalty on adjusted poses")
    parser.add_argument("--weight_pose_reg", type=float, default=0.1)
    parser.add_argument("--weight_coplanar", type=float, default=0.1)
    parser.add_argument("--colinear", action="store_true", help="add colinear penalty to coplanar regularization")
    parser.add_argument("--pose_lr", type=float, default=9e-4, help="LR for pose refinement")
    parser.add_argument("--schedule_pose", action="store_true", help="toggle LR scheduling for pose refinment")

    # Demo
    parser.add_argument("--demo", action="store_true", help="save demo checkpoint")
    parser.add_argument("--demo_name", type=str, default=None, help="name of demo .pth file")

    return parser

def write_args(args):
    f = Path(args.workspace) / "args"
    f.mkdir(exist_ok=True, parents=True)
    f = f / f"{args.name}_args.txt"
    with open(f, "w") as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
