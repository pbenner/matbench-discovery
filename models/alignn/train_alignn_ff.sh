#! /bin/bash

python ~/Source/tmp/alignn/alignn/train_folder_ff.py --root_dir data-train-ff --output_dir data-train-ff-result --config_name alignn-ff-config.json > train_alignn_ff.log 2>&1 & disown

