#! /bin/sh

INFILE_TRAIN=data-train.xyz
INFILE_VALID=data-valid.xyz

for r in 3.5 4.5 5.0 5.5 6.0; do

    echo "Processing r=${r}..."

    OUTDIR=data-r${r}

    rm -rf ${OUTDIR} && mkdir ${OUTDIR}

    python ./mace/scripts/preprocess_data.py \
	   --train_file="${INFILE_TRAIN}" \
	   --valid_file="${INFILE_VALID}" \
	   --atomic_numbers="[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 89, 90, 91, 92, 93, 94]" \
	   --r_max=${r} \
	   --h5_prefix="${OUTDIR}/" \
	   --compute_statistics \
	   --E0s="average" \
	   --seed=42

done
