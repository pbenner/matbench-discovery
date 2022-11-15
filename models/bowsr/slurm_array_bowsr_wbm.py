# %%
from __future__ import annotations

import contextlib
import os
from datetime import datetime
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
import wandb
from maml.apps.bowsr.model.megnet import MEGNet
from maml.apps.bowsr.optimizer import BayesianOptimizer
from tqdm import tqdm

from matbench_discovery import ROOT, as_dict_handler
from matbench_discovery.slurm import slurm_submit_python

__author__ = "Janosh Riebesell"
__date__ = "2022-08-15"

"""
To slurm submit this file: python path/to/file.py slurm-submit

Requires MEGNet and MAML installation: pip install megnet maml
"""

task_type = "IS2RE"  # "RS2RE"
module_dir = os.path.dirname(__file__)
# --mem 12000 avoids slurmstepd: error: Detected 1 oom-kill event(s)
#     Some of your processes may have been killed by the cgroup out-of-memory handler.
slurm_mem_per_node = 12000
# set large job array size for fast testing/debugging
slurm_array_task_count = 500
timestamp = f"{datetime.now():%Y-%m-%d@%H-%M-%S}"
today = timestamp.split("@")[0]
slurm_job_id = os.environ.get("SLURM_JOB_ID", "debug")
job_name = f"bowsr-megnet-wbm-{task_type}-{slurm_job_id}"
out_dir = f"{module_dir}/{today}-{job_name}"

data_path = f"{ROOT}/data/2022-10-19-wbm-init-structs.json.gz"

slurm_vars = slurm_submit_python(
    job_name=job_name,
    log_dir=out_dir,
    partition="icelake-himem",
    account="LEE-SL3-CPU",
    time=(slurm_max_job_time := "3:0:0"),
    # --time 2h is probably enough but best be safe.
    array=f"1-{slurm_array_task_count}",
    slurm_flags=("--mem", str(slurm_mem_per_node)),
    # TF_CPP_MIN_LOG_LEVEL=2 means INFO and WARNING logs are not printed
    # https://stackoverflow.com/a/40982782
    pre_cmd="TF_CPP_MIN_LOG_LEVEL=2",
)


# %%
slurm_array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
out_path = f"{out_dir}/{slurm_array_task_id}.json.gz"

print(f"Job started running {timestamp}")
print(f"{data_path = }")
print(f"{out_path = }")
print(f"{version('maml') = }")
print(f"{version('megnet') = }")


if os.path.isfile(out_path):
    raise SystemExit(f"{out_path = } already exists, exciting early")


# %%
print(f"Loading from {data_path = }")
df_wbm = pd.read_json(data_path).set_index("material_id")

df_this_job: pd.DataFrame = np.array_split(df_wbm, slurm_array_task_count)[
    slurm_array_task_id - 1
]


# %%
bayes_optim_kwargs = dict(
    relax_coords=True,
    relax_lattice=True,
    use_symmetry=True,
    seed=42,
)
optimize_kwargs = dict(n_init=100, n_iter=100, alpha=0.026**2)

run_params = dict(
    bayes_optim_kwargs=bayes_optim_kwargs,
    data_path=data_path,
    df=dict(shape=str(df_this_job.shape), columns=", ".join(df_this_job)),
    maml_version=version("maml"),
    megnet_version=version("megnet"),
    optimize_kwargs=optimize_kwargs,
    task_type=task_type,
    slurm_array_task_count=slurm_array_task_count,
    slurm_max_job_time=slurm_max_job_time,
    **slurm_vars,
)
if wandb.run is None:
    wandb.login()

# getting wandb: 429 encountered ({"error":"rate limit exceeded"}), retrying request
# https://community.wandb.ai/t/753/14
wandb.init(
    entity="janosh",
    project="matbench-discovery",
    name=f"{job_name}-{slurm_array_task_id}",
    config=run_params,
)


# %%
model = MEGNet()
relax_results: dict[str, dict[str, Any]] = {}

if task_type == "IS2RE":
    from pymatgen.core import Structure

    structures = df_this_job.initial_structure.map(Structure.from_dict)
elif task_type == "RS2RE":
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    structures = df_this_job.cse.map(
        lambda x: ComputedStructureEntry.from_dict(x).structure
    )
else:
    raise ValueError(f"Unknown {task_type = }")


for material_id, structure in tqdm(
    structures.items(), desc="Main loop", total=len(structures), disable=None
):
    if material_id in relax_results:
        continue
    bayes_optimizer = BayesianOptimizer(
        model=model, structure=structure, **bayes_optim_kwargs
    )
    bayes_optimizer.set_bounds()
    # reason for devnull here: https://github.com/materialsvirtuallab/maml/issues/469
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        bayes_optimizer.optimize(**optimize_kwargs)

    structure_bowsr, energy_bowsr = bayes_optimizer.get_optimized_structure_and_energy()

    results = dict(
        e_form_per_atom_bowsr=model.predict_energy(structure),
        structure_bowsr=structure_bowsr,
        energy_bowsr=energy_bowsr,
    )

    relax_results[material_id] = results


# %%
df_output = pd.DataFrame(relax_results).T
df_output.index.name = "material_id"

df_output.reset_index().to_json(out_path, default_handler=as_dict_handler)

wandb.log_artifact(out_path, type=job_name)
