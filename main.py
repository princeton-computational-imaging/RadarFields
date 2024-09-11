from pathlib import Path
import copy

import torch
import numpy as np

from parse import get_arg_parser, write_args
from radarfields.dataset import RadarDataset
from radarfields.nn.models import RadarField
from radarfields.train import Trainer
from utils.data import filter_dict_for_dataclass
from utils.train import seed_everything

loss_dict = {
        "mse": torch.nn.MSELoss(),
        "l1": torch.nn.L1Loss(),
        "kl": torch.nn.KLDivLoss(reduction="batchmean")
}

def test(args, model, criterion):

    test_loader = RadarDataset(split="test",
                                **filter_dict_for_dataclass(RadarDataset, vars(copy.deepcopy(args))),
                                **args.intrinsics_radar
                                ).dataloader(args.bs)
    print("Test dataloader prepared.")

    # Tracking sensor poses
    args.all_poses = test_loader._data.poses_radar.to(args.device)
    args.test_indices = test_loader._data.preprocess["test_indices"]

    trainer = Trainer(args, model, split="test", criterion=criterion,
                      optimizer=None, lr_scheduler=None,device=args.device)

    trainer.test(test_loader)

def train(args, model, criterion):

    optimizer = lambda model: torch.optim.Adam(
        model.get_params(args.lr), betas=(0.9, 0.99), eps=1e-15
    )
    scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda iter: 0.1 ** min(iter / args.iters, 1)
    ) # (decay to 0.1 * init_lr at last iter step)

    train_loader = RadarDataset(split="train",
                                **filter_dict_for_dataclass(RadarDataset, vars(copy.deepcopy(args))),
                                **args.intrinsics_radar
                                ).dataloader(args.bs)
    print("Train dataloader prepared.")

    # Tracking sensor poses
    args.all_poses = train_loader._data.poses_radar.to(args.device)
    args.test_indices = train_loader._data.preprocess["test_indices"]

    trainer = Trainer(args, model, split="train", criterion=criterion,
                      optimizer=optimizer, lr_scheduler=scheduler, device=args.device)

    # Training
    max_epoch = np.ceil(args.iters / len(train_loader)).astype(np.int32)
    print(f"max_epoch: {max_epoch}")
    trainer.train(train_loader, max_epoch)

    # Freeing memory
    del train_loader
    del trainer
    torch.cuda.empty_cache()

    # Testing
    if not args.skip_test:
        test(args, model, criterion)

def main():
    # Get config args
    parser = get_arg_parser()
    args = parser.parse_args()

    seed_everything(args.seed)

    args.data_path = Path(__file__).parent / 'data' / args.seq
    print(f"Reading training data from: {args.data_path}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    write_args(args) # Writing config args to .txt file
    model = RadarField(**args.model_settings, use_tcnn=args.tcnn)
    if args.tcnn: print("Using TCNN backbone")
    
    criterion = {
        "fft": loss_dict[args.fft_loss.strip("\"")],
        "occ": loss_dict[args.occ_loss.strip("\"")]
    }

    if args.test: test(args, model, criterion)
    else: train(args, model, criterion)

if __name__ == "__main__":
    main()