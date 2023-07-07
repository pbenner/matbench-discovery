# %%
from __future__ import annotations

import json
import os
from importlib.metadata import version

import numpy  as np
import pandas as pd
import torch
import gzip

from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN
from alignn.pretrained import all_models, get_figshare_model
from alignn.ff.ff import ForceField, default_path
from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from matbench_discovery import DEBUG, today
from matbench_discovery.data import DATA_FILES, df_wbm
# %%

__author__ = "Janosh Riebesell, Philipp Benner"
__date__ = "2023-06-03"

# %% Environment variables

task_id = int(os.getenv("TASK_ID", default="0"))
out_dir = os.getenv("SBATCH_OUTPUT", default='2022-10-19-wbm-alignn-relaxed-structs')

# %%
n_splits = 100
module_dir = os.path.dirname(__file__)
# model_name = "mp_e_form_alignn"  # pre-trained by NIST
model_name = f"{module_dir}/data-train-result/best-model.pth"
task_type = "IS2RE"
target_col = "e_form_per_atom_mp2020_corrected"
input_col = "initial_structure"
id_col = "material_id"
device = f'cuda:{task_id%4}' if torch.cuda.is_available() else "cpu"
job_name = f"{model_name}-wbm-{task_type}{'-debug' if DEBUG else ''}"
if task_id == 0:
    out_path = '2022-10-19-wbm-alignn-relaxed-structs.json.gz'
else:
    out_path = f'{out_dir}/batch-{task_id}.json.gz'

if task_id < 0 or task_id > n_splits:
    raise SystemExit(f"Invalid task_id={task_id}")
if task_id > 0 and not os.path.exists(out_dir):
    os.mkdir(out_dir)
if os.path.isfile(out_path):
    raise SystemExit(f"{out_path = } already exists, exiting")

# %% Load data
data_path = {
    "IS2RE": DATA_FILES.wbm_initial_structures,
    "RS2RE": DATA_FILES.wbm_computed_structure_entries,
}[task_type]
input_col = {"IS2RE": "initial_structure", "RS2RE": "relaxed_structure"}[task_type]

df_in = pd.read_json(data_path).set_index(id_col)

df_in[target_col] = df_wbm[target_col]
if task_type == "RS2RE":
    df_in[input_col] = [x["structure"] for x in df_in.computed_structure_entry]
assert input_col in df_in, f"{input_col=} not in {list(df_in)}"

# Split data into parts and process only one batch
if task_id != 0:
    df_in = np.array_split(df_in, 100)[task_id-1]
    print(f'Relaxing materials in range {df_in.index[0]} - {df_in.index[-1]}')
else:
    print(f'Relaxing full range of materials')

# %% Relax structures

def alignn_relax(structure, ff_model_path=default_path()):
    ff = ForceField(
        jarvis_atoms=JarvisAtomsAdaptor.get_atoms(Structure.from_dict(structure)),
        model_path=ff_model_path,
        device=device,
        logfile='/dev/null'
    )
    # Relax structure
    opt, _, _ = ff.optimize_atoms(trajectory=None, logfile='/dev/null')

    return JarvisAtomsAdaptor.get_structure(opt).as_dict()

# Convert str
df_relaxed = pd.Series()

for material_id in tqdm(df_in.index):
    df_relaxed[material_id] = alignn_relax(df_in.loc[material_id]['initial_structure'])

df_in = df_in.assign(relaxed_structure = df_relaxed)

# %% Dump result

with gzip.open(out_path, 'wb') as f:
    df_in.to_json(f)

# df_in = pd.read_json('2022-10-19-wbm-alignn-relaxed-structs.json.gz')

# %%
