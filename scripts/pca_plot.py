# %% Imports

import numpy as np
import matplotlib.pyplot as plt

# %%

def plot_scatter(u, y, filename = None):
    plt.scatter(u[:, 0], u[:, 1], c=y, cmap='Spectral', s=5)
    plt.colorbar(boundaries=np.arange(min(y), max(y)+2)-0.5).set_ticks(np.arange(min(y), max(y)+1))
    if filename is not None:
        plt.savefig(filename)
    plt.show()

# %%

filename_in  = f'pca.npz'
filename_out = f'pca.png'

X = np.load(filename_in)['X']
y = np.load(filename_in)['y']

# Plot all data
plot_scatter(X, y, filename = filename_out)

# %%

filename_in  = f'pca_mptrain.npz'
filename_out = f'pca_mptrain.png'

X = np.load(filename_in)['X']
y = np.load(filename_in)['y']

# Plot all data
plot_scatter(X, y, filename = filename_out)

# %%
