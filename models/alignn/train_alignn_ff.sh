#! /bin/bash

python ~/Source/tmp/alignn/alignn/train_folder_ff.py --root_dir data-train-ff --output_dir data-train-ff-result --config_name alignn-ff-config.json --restart_model_path ../../../alignn/alignn/ff/alignnff_wt10/best_model.pt > train_alignn_ff.log 2>&1 & disown

accelerate launch --multi_gpu ~/Source/tmp/alignn/alignn/train_folder_ff.py --root_dir data-train-ff --output_dir data-train-ff-result --config_name alignn-ff-config.json --restart_model_path ../../../alignn/alignn/ff/alignnff_wt10/best_model.pt > train_alignn_ff.log 2>&1 & disown
