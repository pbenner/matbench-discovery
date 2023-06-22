# %%
from __future__ import annotations

import json
import os
from importlib.metadata import version

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

# %%
module_dir = os.path.dirname(__file__)
# model_name = "mp_e_form_alignn"  # pre-trained by NIST
model_name = f"{module_dir}/data-train-result/best-model.pth"
task_type = "IS2RE"
target_col = "e_form_per_atom_mp2020_corrected"
input_col = "initial_structure"
id_col = "material_id"
device = "cuda" if torch.cuda.is_available() else "cpu"
job_name = f"{model_name}-wbm-{task_type}{'-debug' if DEBUG else ''}"
out_dir = os.getenv("SBATCH_OUTPUT", f"{module_dir}/{today}-{job_name}")

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

# %% Relax structures

def alignn_relax(x, ff_model_path=default_path()):
    material_id, structure = x

    ff = ForceField(
        jarvis_atoms=JarvisAtomsAdaptor.get_atoms(Structure.from_dict(structure)),
        model_path=ff_model_path,
        device='cuda:2'
    )
    # Relax structure
    opt, _, _ = ff.optimize_atoms()

    return material_id, JarvisAtomsAdaptor.get_structure(opt).as_dict()

# Compute relaxations
r = thread_map(
    alignn_relax,
    df_in[input_col].items(),
    chunksize=64,
    desc="Relaxing structures",
    colour='green',
    max_workers=2)

r = pd.Series(index = [ x[0] for x in r ], data = [ x[1] for x in r ])

df_in = df_in.assign(relaxed_structure = r)

# %% Dump result

with gzip.open('2022-10-19-wbm-alignn-relaxed-structs.json.gz', 'wb') as f:
    df_in.to_json(f)

# df_in = pd.read_json('2022-10-19-wbm-alignn-relaxed-structs.json.gz')

# %%
