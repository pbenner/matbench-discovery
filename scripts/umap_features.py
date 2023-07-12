# %% Imports

import numpy as np
import pandas as pd

from matbench_discovery.data import DATA_FILES, df_wbm
from pymatgen.core import Structure
from tqdm import tqdm

# %% Filter alloys of interest (AoI)

def filter_aoi_f(structure) -> bool:
    for element in structure.composition.elements:
        if not element.is_metal:
            return False
        if element.number > 56:
            return False
    return True

# %% Material Structure and Composition Featurizer
# "A critical examination of robustness and generalizability of machine learning prediction
#  of materials properties"](https://www.nature.com/articles/s41524-023-01012-9) by
# Kangming Li, Brian DeCost, Kamal Choudhary, Michael Greenwood, and Jason Hattrick-Simpers.

def to_unitcell(structure):
    [site.to_unit_cell(in_place=True) for site in structure.sites]
    return structure

def featurize_dataframe(
        df_in,
        col_id='structure',
        ignore_errors=True,
        chunksize=30
        ):
    """
    Featurize a dataframe using Matminter Structure featurizer

    Parameters
    ----------
    df : Pandas.DataFrame 
        DataFrame with a column named "structure"

    Returns
    -------
    A DataFrame containing 273 features (columns)

    """
    # For featurization
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (ElementProperty, 
                                                  Stoichiometry, 
                                                  ValenceOrbital, 
                                                  IonProperty)
    from matminer.featurizers.structure import (SiteStatsFingerprint, 
                                                StructuralHeterogeneity,
                                                ChemicalOrdering, 
                                                StructureComposition, 
                                                MaximumPackingEfficiency)
    
    
    if isinstance(df_in, pd.Series):
        df = df_in.to_frame()
    else:
        df = df_in
    df[col_id] = df[col_id].apply(to_unitcell)
    
    # 128 structural feature
    struc_feat = [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"), 
        SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
        StructuralHeterogeneity(),
        MaximumPackingEfficiency(),
        ChemicalOrdering()
        ]       
    # 145 compositional features
    compo_feat = [
        StructureComposition(Stoichiometry()),
        StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=['frac'])),
        StructureComposition(IonProperty(fast=True))
        ]
    featurizer = MultipleFeaturizer(struc_feat+compo_feat)    
    # Set the chunksize used for Pool.map parallelisation
    featurizer.set_chunksize(chunksize=chunksize)
    featurizer.fit(df[col_id])
    X = featurizer.featurize_dataframe(df=df,col_id=col_id,ignore_errors=ignore_errors)  
    # check failed entries    
    print('Featurization completed.')
    failed = np.any(pd.isnull(X.iloc[:,df.shape[1]:]), axis=1)
    if np.sum(failed) > 0:
        print(f'Number failed: {np.sum(failed)}/{len(failed)}')
    return X

# %%

def featurize_file(filename, computed_structure = False, filter_aoi = False, input_col = "initial_structure", id_col = "material_id"):

    df_in = pd.read_json(filename).set_index(id_col)

    if computed_structure:
        df_in[input_col] = [
            Structure.from_dict(x['structure'])
            for x in tqdm(df_in[input_col], leave=False, desc="Converting to PyMatgen Structure") ]
    else:
        df_in[input_col] = [
            Structure.from_dict(x)
            for x in tqdm(df_in[input_col], leave=False, desc="Converting to PyMatgen Structure") ]

    if filter_aoi:
        df_in = df_in[df_in[input_col].apply(filter_aoi_f)]

    df_features = featurize_dataframe(df_in[input_col], col_id=input_col)

    return df_features.drop(input_col, axis=1)

# %% Compute features and export to CSV

df_features = featurize_file(DATA_FILES.wbm_initial_structures)
df_features.to_csv('2022-10-19-wbm-init-structs-features.csv.bz2')

df_features = featurize_file(DATA_FILES.wbm_computed_structure_entries, input_col = 'computed_structure_entry', computed_structure = True)
df_features.to_csv('2022-10-19-wbm-computed-structure-entries-features.csv.bz2')

df_features = featurize_file(DATA_FILES.mp_computed_structure_entries, input_col = 'entry', computed_structure = True)
df_features.to_csv('2023-02-07-mp-computed-structure-entries-features.csv.bz2')


# %%

df_features = featurize_file(DATA_FILES.wbm_initial_structures, filter_aoi = True)
df_features.to_csv('2022-10-19-wbm-init-structs-features-aoi.csv.bz2')

df_features = featurize_file(DATA_FILES.wbm_computed_structure_entries, filter_aoi = True, input_col = 'computed_structure_entry', computed_structure = True)
df_features.to_csv('2022-10-19-wbm-computed-structure-entries-features-aoi.csv.bz2')

df_features = featurize_file(DATA_FILES.mp_computed_structure_entries, filter_aoi = True, input_col = 'entry', computed_structure = True)
df_features.to_csv('2023-02-07-mp-computed-structure-entries-features-aoi.csv.bz2')

# %%
