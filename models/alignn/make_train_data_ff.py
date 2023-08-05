# %% Imports
import json
import os

from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from tqdm import tqdm

# %%

__author__ = "Philipp Benner"
__date__ = "2023-06-02"

# %%

outdir = 'data-train-ff'

# %%

with open('MPtrj_2022.9_full.json') as f:
    js = json.load(f)

r = []
for _, values in tqdm(js.items(), desc='Converting data', total=len(js)):
    for submid, subvalues in values.items():
        atoms = JarvisAtomsAdaptor.get_atoms(Structure.from_dict(subvalues['structure']))
        r.append({
            'jid'         : submid,
            'total_energy': subvalues['corrected_total_energy'],
            'atoms'       : json.dumps(atoms.to_dict()),
            'forces'      : subvalues['force'],
            'stresses'    : subvalues['stress'],
        })

if not os.path.exists(outdir):
    os.makedirs(outdir)

# json.dump exports atoms dict as string, do manual export instead
with open(f'{outdir}/id_prop.json', 'w') as f:
    _ = f.write('[')
    for i, item in enumerate(r):
        if i > 0:
            _ = f.write(', ')
        _ = f.write('{'+f'"jid": "{item["jid"]}", "total_energy": {item["total_energy"]}, "atoms": {item["atoms"]}, "forces": {item["forces"]}, "stresses": {item["stresses"]}'+'}')
    _ = f.write(']\n')
