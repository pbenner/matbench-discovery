## ALIGNN formation energy predictions on WBM test set

ALIGNN was trained for 1000 epochs using L1 loss. The model that performed best on the validation set was [uploaded to Figshare](https://figshare.com/account/articles/22715158?file=41233560) as `2023-06-02-pbenner-best-alignn-model.pth.zip` and used for predictions. This required minor changes to the ALIGNN source code provided in `alignn-2023.01.10.patch`

1. Fix use without test set (see [ALIGNN #104)](https://github.com/usnistgov/alignn/issues/104#issue-1723978225). In this case, we forked a test set, but it might be better to use the entire data, as mentioned above, especially if the test set by chance contains some important outliers.
1. The `Checkpoint` handler in ALIGNN does not define a score name (see [`train.py`](https://github.com/usnistgov/alignn/blob/46334500cac9833125b3e444d65d0246e692bd61/alignn/train.py#L851)), so it will just save the last two models during training. With this patch, also the best model in terms of accuracy on the validation set is saved, which is the one used to make predictions. This is important, because I used a relatively large `n_early_stopping` in case the validation accuracy shows a double descent (see [Figure 10](https://arxiv.org/pdf/1912.02292.pdf)).

The changes in `alignn-2023.01.10.patch` were applied to ALIGNN version `2023.01.10`.

To reproduce the `alignn` package state used for this submission, run

```bash
pip install alignn==2023.01.10
alignn_dir=$(python -c "import alignn; print(alignn.__path__[0])")
cd $alignn_dir
git apply /path/to/alignn-2023.01.10.patch
```

Replace `/path/to/` with the actual path to the patch file.

The directory contains the following files, which must be executed in the given order to reproduce the results:

1. `train_data.py`: Export Matbench Discovery training data to ALIGNN compatible format. This script outputs training data in the directory `data_train`. In addition, a small test data set is set apart and stored in the directory `data_test`
1. `train_alignn.py`: Train an ALIGNN model on previously exported data. The resulting model is stored in the directory `data-train-result`
1. `test_alignn.py`: Test a trained ALIGNN model on the WBM data. Generates `2023-06-03-mp-e-form-alignn-wbm-IS2RE.csv.gz`.

## ALIGNN-FF formation energy predictions on relaxed WBM test set

The patch `alignn-ff-2023.07.05.patch` fixes the following issue:
```bash
Traceback (most recent call last):
  File "alignn_relax.py", line 96, in <module>
  File "alignn_relax.py", line 88, in alignn_relax
  File "../alignn/ff/ff.py", line 310, in optimize_atoms
  File "../alignn/lib/python3.9/site-packages/ase/optimize/optimize.py", line 269, in run
  File "../alignn/lib/python3.9/site-packages/ase/optimize/optimize.py", line 156, in run
  File "../alignn/lib/python3.9/site-packages/ase/optimize/optimize.py", line 129, in irun
  File "../alignn/lib/python3.9/site-packages/ase/optimize/optimize.py", line 108, in call_observers
  File "../alignn/lib/python3.9/site-packages/ase/io/trajectory.py", line 132, in write
  File "../alignn/lib/python3.9/site-packages/ase/io/trajectory.py", line 156, in _write_atoms
  File "../alignn/lib/python3.9/site-packages/ase/io/trajectory.py", line 381, in write_atoms
  File "../alignn/lib/python3.9/site-packages/ase/io/ulm.py", line 400, in write
  File "../alignn/lib/python3.9/site-packages/ase/io/ulm.py", line 325, in fill
OSError: [Errno 24] Too many open files
```

To reproduce the ALIGNN relaxed predictions, run the following scripts:
1. `alignn_relax.py`: Set the variable `n_splits` to the number of GPU compute nodes. On each compute node, set the environment variable `TASK_ID` to a value in the range 1-`n_splits`. Set the variable `n_processes_per_task` to the number of processes on a single node. For 48 cpu cores with 4 GPUs a good setting is to use 10 processes.
2. `test_alignn_relaxed.py`: Read the relaxed structures and compute predictions. Set the variable `n_splits` accordingly.