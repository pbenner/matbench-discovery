#! /bin/sh

BASEDIR=data-train-preprocessed
OUTDIR=data-train-result-stress-128-inter3

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
    --num_interactions=3 \
    --num_channels=128 \
    --max_L=1 \
    --correlation=3 \
    --batch_size=32 \
    --valid_batch_size=32 \
    --max_num_epochs=110 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --loss='stress-l1' \
    --error_table='PerAtomMAEstress' \
    --seed=42 \
    --energy_weight=5.0 \
    --forces_weight=1.0 \
    --stress_weight=500.0 \
    --compute_stress True \
    --restart_latest \
    --restart_lr=0.00008 \
    --save_cpu
