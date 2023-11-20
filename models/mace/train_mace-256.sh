#! /bin/sh

BASEDIR=data-train-preprocessed
OUTDIR=data-train-result-256

mkdir -p ${OUTDIR}

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
    --num_workers=8 \
    --model="ScaleShiftMACE" \
    --num_interactions=2 \
    --num_channels=256 \
    --max_L=1 \
    --correlation=3 \
    --batch_size=32 \
    --valid_batch_size=32 \
    --max_num_epochs=200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --loss='weighted-l1' \
    --error_table='PerAtomMAE' \
    --seed=42 \
    --forces_weight=1.0 \
    --energy_weight=1.0 \
    --restart_latest \
    --restart_lr=0.003 \
    --save_cpu
