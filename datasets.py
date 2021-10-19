import torch
from torch_geometric.data import DataLoader
from torch_cmspepr.dataset import TauDataset

def tau_dataset(root='data/taus'):
    dataset = TauDataset(root)
    dataset.blacklist([ # Remove a bunch of bad events
        'data/taus/110_nanoML_98.npz',
        'data/taus/113_nanoML_13.npz',
        'data/taus/124_nanoML_77.npz',
        'data/taus/128_nanoML_70.npz',
        'data/taus/149_nanoML_90.npz',
        'data/taus/153_nanoML_22.npz',
        'data/taus/26_nanoML_93.npz',
        'data/taus/32_nanoML_45.npz',
        'data/taus/5_nanoML_51.npz',
        'data/taus/86_nanoML_97.npz',
        ])
    return dataset
