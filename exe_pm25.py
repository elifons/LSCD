import argparse
import torch
import datetime
import json
import yaml
import os
import sys
import wandb
from functools import partial

from dataset_pm25 import get_dataloader
from main_model import *
from utils import train, evaluate

parser = argparse.ArgumentParser(description="PM25")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--targetstrategy", type=str, default="mix", choices=["mix", "random", "historical"])
parser.add_argument("--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--nrows", type=int, default=1)
parser.add_argument("--ncols", type=int, default=3)
parser.add_argument("--figsize", default=(14,3))
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--wandb_name", type=str, default="pm25_runs_all")
parser.add_argument("--eval_at", nargs="+", type=int, default=[-1])
parser.add_argument('--epochs', type=int, default=-1)
parser.add_argument('--ls_log', action='store_true', default=False, help='apply log to ls')
parser.add_argument('--ls_sqrt', action='store_true', default=False, help='apply sqrt to ls')
parser.add_argument('--ls_pe', action='store_true', default=False, help='apply pe to ls')
parser.add_argument('--ls_nheads', type=int, default=1, help='number of heads for ls')
parser.add_argument('--ls_nlayers', type=int, default=1, help='number of layers for ls')
parser.add_argument('--ls_channels', type=int, default=1, help='number of channels for ls. must be divisisble by ls_nheads.')
parser.add_argument('--closs_w', type=float, default=1.0)
parser.add_argument('--closs_start', type=int, default=sys.maxsize)
args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.epochs > 0:
	config["train"]["epochs"] = args.epochs
config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy

config["model"]["ls_log"] = args.ls_log
config["model"]["ls_sqrt"] = args.ls_sqrt
config["model"]["ls_pe"] = args.ls_pe
config["model"]["ls_nheads"] = args.ls_nheads
config["model"]["ls_nlayers"] = args.ls_nlayers
config["model"]["ls_channels"] = args.ls_channels

config["train"]["closs_w"] = args.closs_w
config["train"]["closs_start"] = args.closs_start

print(json.dumps(config, indent=4))

args_dict = vars(args)
full_config = {**config, **args_dict}

# Initialize WandB with the full configuration
wandb.init(
    project=args.wandb_name,  # Replace with your project name
    config=full_config
)

# Access parameters in WandB, if needed
print(wandb.config)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
foldername = ("./save/pm25_validationindex" + str(args.validationindex) + "_" + current_time + "/")
wandb.run.summary["model_folder"] = foldername
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"], device=args.device, validindex=args.validationindex
)

t = train_loader.dataset[0]['timepoints']

if args.model in models_dict:
	model = models_dict[args.model](config=config, device=args.device, target_dim=36, timesteps=t, dataset="PM25").to(args.device)
else:
	raise ValueError("Invalid model name")

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        wandb=wandb,
        eval_at=args.eval_at,
        test_loader=test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/best_model.pth"))

plot_params = {
    "nrows": args.nrows,
    "ncols": args.ncols,
    "figsize": args.figsize,
}

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    plot_params=plot_params, 
    wandb=wandb, 
    model_name=args.model)
wandb.finish()
