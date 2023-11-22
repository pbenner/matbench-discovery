# %% Imports
import json
import numpy as np

from ase.io import write
from ase.units import GPa
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

# %%

__author__ = "Philipp Benner"
__date__ = "2023-08-11"

# %%

# Get data from Figshare:
# https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842

with open('MPtrj_2022.9_full.json') as f:
    js = json.load(f)

r = []
for _, values in tqdm(js.items(), desc='Converting data', total=len(js)):
    for submid, subvalues in values.items():
        atoms = AseAtomsAdaptor.get_atoms(
            Structure.from_dict(subvalues['structure']),
            info={'config_type'     : 'Default',
                  'energy'          : subvalues['uncorrected_total_energy'],
                  'energy_corrected': subvalues['corrected_total_energy'],
                  'stress'          : np.array(subvalues['stress']) * 1e-1 * GPa })
        atoms.arrays['forces'] = np.array(subvalues['force'])

        r.append(atoms)

# %% Export result

write('data-train.xyz', r)
