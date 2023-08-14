#! /bin/sh

BASEDIR=data-train-preprocessed
OUTDIR=data-train-result

mkdir -p ${data-train-result}

accelerate launch --multi_gpu --num_processes=4 ./mace/scripts/run_train.py \
    --name="MACE_MPtrj_2022.9" \
    --log_dir="${OUTDIR}/logs" \
    --model_dir="${OUTDIR}" \
    --checkpoints_dir="${OUTDIR}/checkpoints" \
    --results_dir="${OUTDIR}/results" \
    --downloads_dir="${OUTDIR}/downloads" \
    --train_file="./${BASEDIR}/train.h5" \
    --valid_file="./${BASEDIR}/valid.h5" \
    --statistics_file="./${BASEDIR}/statistics.json" \
    --accelerate \
    --num_workers=16 \
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
