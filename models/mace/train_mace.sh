#! /bin/sh

BASEDIR=data-train-preprocessed

python ./mace/scripts/run_train.py \
    --name="MACE_MPtrj_2022.9" \
    --num_workers=16 \
    --train_file="./${BASEDIR}/train.h5" \
    --valid_file="./${BASEDIR}/valid.h5" \
    --statistics_file="./${BASEDIR}/statistics.json" \
    --model="ScaleShiftMACE" \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --batch_size=32 \
    --valid_batch_size=32 \
    --max_num_epochs=100 \
    --swa \
    --start_swa=60 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --error_table='PerAtomMAE' \
    --device=cuda \
    --seed=42
