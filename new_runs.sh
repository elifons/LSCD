#!/bin/bash


python exe_sines.py --model csdi_sidels --testmissingratio 0.1 --seed 1 
python exe_sines.py --model csdi_sidels --testmissingratio 0.1 --seed 1 --missing_type seq
python exe_sines.py --model csdi_sidels --testmissingratio 0.5 --seed 1 
python exe_sines.py --model csdi_sidels --testmissingratio 0.5 --seed 1 --missing_type seq
python exe_sines.py --model csdi_sidels --testmissingratio 0.9 --seed 1 
python exe_sines.py --model csdi_sidels --testmissingratio 0.9 --seed 1 --missing_type seq

missing_rates=(0.1 0.5 0.9)
folds=(0)
#models=("csdi_sidels_lsenc") # "csdi_sidels" "csdi_dual" "csdi_sidels_consloss")
missing_types=("point" "seq" "block")
# Loop through each combination of model, missing rate, and seed
for fold in "${folds[@]}"; do
    for missing_type in "${missing_types[@]}"; do
        for missing_rate in "${missing_rates[@]}"; do
            echo "Running: python exe_sines.py --model csdi_sidels_lsenc --seed 1 --wandb_name full_sines_2000 --epochs 400 --ls_log --ls_nheads 8 --ls_nlayers 4 --ls_channels 64 --nfold $fold --testmissingratio $missing_rate --missing_type $missing_type"
            python exe_sines.py --model csdi_sidels_lsenc --seed 1 --wandb_name full_sines_2000 --epochs 400 --ls_log --ls_nheads 8 --ls_nlayers 4 --ls_channels 64 --nfold "$fold" --testmissingratio "$missing_rate" --missing_type "$missing_type"
        done
    done
done


