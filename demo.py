from pathlib import Path

import torch

from parse import get_arg_parser
from radarfields.nn.models import RadarField
from radarfields.train import Trainer
from utils.train import seed_everything

loss_dict = {
        "mse": torch.nn.MSELoss(),
        "l1": torch.nn.L1Loss(),
        "kl": torch.nn.KLDivLoss(reduction="batchmean")
}

def main():
    # Get config args
    parser = get_arg_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    args.name = args.demo_name

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RadarField(**args.model_settings, use_tcnn=args.tcnn)
    if args.tcnn: print("Using TCNN backbone")
    
    criterion = {
        "fft": loss_dict[args.fft_loss.strip("\"")],
        "occ": loss_dict[args.occ_loss.strip("\"")]
    }

    trainer = Trainer(args, model, split="test", criterion=criterion,
                      optimizer=None, lr_scheduler=None, device=args.device, skip_ckpt=True)
    loader = trainer.load_checkpoint(checkpoint=Path("checkpoints") / (args.demo_name + '.pth'), demo=True)
    args.all_poses = loader._data.poses_radar.to(args.device)
    args.test_indices = loader._data.preprocess["test_indices"]
    trainer.test(loader)

if __name__ == "__main__":
    main()
