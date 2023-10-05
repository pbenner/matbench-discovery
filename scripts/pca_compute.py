# %% Imports

import numpy as np
import pandas as pd
import re
import os

from sklearn.decomposition import PCA

# %%

def select_features(df_features, threshold = 0.7):

    corr_matrix = df_features.corr(method='pearson', numeric_only=False).abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than our threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features 
    return df_features.drop(to_drop, axis=1)

def import_features(filename, id_col = "material_id"):
    df = pd.read_csv(filename)
    df = df.set_index(id_col)
    return df

# %%

df_in = import_features('2022-10-19-wbm-init-structs-features.csv.bz2')
df_mp = import_features('2023-02-07-mp-computed-structure-entries-features.csv.bz2')
df    = pd.concat((df_in, df_mp))

# Drop all rows containing NaN values
df     = df   .dropna(axis=0)
df_mp  = df_mp.dropna(axis=0)

# Drop highly correlated features
df    = select_features(df)
df_mp = df_mp[df.columns]

# %% Create labels

y = [ re.sub(r'^wbm-(\d+)-\d+$', r'\1', id) for id in df.index ]
y = [ re.sub(r'^(?:mp|mvc)-\d+$', r'0', id) for id in y ]
y = pd.Series(y, index=df.index).astype(int).to_numpy()

y_mp = [ re.sub(r'^(?:mp|mvc)-\d+$', r'0', id) for id in df_mp.index ]
y_mp = pd.Series(y_mp, index=df_mp.index).astype(int).to_numpy()

# %%

pca = PCA(n_components=2)
pca.fit(df)

X = pca.transform(df)

np.savez('pca.npz', X = X, y = y)

# %%

pca = PCA(n_components=2)
pca.fit(df_mp)

X = pca.transform(df)

np.savez('pca_mptrain.npz', X = X, y = y)

# %%
