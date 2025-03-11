import argparse
import torch
import datetime
import json
import yaml
import os
import sys
import wandb
from functools import partial

from main_model import *
from dataset_sines import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="Sines")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--missing_type", type=str, default="point")
parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--nrows", type=int, default=1)
parser.add_argument("--ncols", type=int, default=5)
parser.add_argument("--figsize", default=(16,3))
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--wandb_name", type=str, default="full_sines_2000")
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
config["model"]["test_missing_ratio"] = args.testmissingratio
config["model"]["missing_type"] = args.missing_type

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

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
# foldername = "./save/sines_fold" + str(args.nfold) + "_" + current_time + "_"  "/"
foldername = f"./save/sines_fold{args.nfold}_{current_time}_{args.model}_{args.missing_type}_{args.testmissingratio}/"
wandb.run.summary["model_folder"] = foldername
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
	json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
	seed=args.seed,
	nfold=args.nfold,
	batch_size=config["train"]["batch_size"],
	missing_ratio=config["model"]["test_missing_ratio"],
	pattern=config["model"]["missing_type"]
)

t = train_loader.dataset[0]['timepoints']

if args.model in models_dict:
	model = models_dict[args.model](config=config, device=args.device, target_dim=5, timesteps=t, dataset="Sines").to(args.device)
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
        model_name=args.model,
        eval_at=args.eval_at,
        test_loader=test_loader,
		nsample=args.nsample
	)
else:
	model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

plot_params = {
	"nrows": args.nrows,
	"ncols": args.ncols,
	"figsize": args.figsize,
}
evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername, plot_params=plot_params, wandb=wandb, model_name=args.model)
wandb.finish()
