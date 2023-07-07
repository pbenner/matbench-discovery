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

for n_neighbors in [5, 15, 75, 150]:

    filename_in   = f'umap_n{n_neighbors}.npz'
    filename_out1 = f'umap_n{n_neighbors}.png'
    filename_out2 = f'umap_n{n_neighbors}_nomp.png'

    u = np.load(filename_in)['u']
    y = np.load(filename_in)['y']

    # Plot all data
    plot_scatter(u, y, filename = filename_out1)
    # Drop MP points
    plot_scatter(u[y > 0], y[y > 0], filename = filename_out2)

# %%

for n_neighbors in [5, 15, 75, 150]:

    filename_in   = f'umap_mptrain_n{n_neighbors}.npz'
    filename_out1 = f'umap_mptrain_n{n_neighbors}.png'

    u = np.load(filename_in)['u']
    #y = np.load(filename_in)['y']

    # Plot all data
    plot_scatter(u, y, filename = filename_out1)

# %%
