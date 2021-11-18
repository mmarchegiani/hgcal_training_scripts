import numpy as np
import torch
from torch_geometric.data import DataLoader, Data
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


def single_photon_dataset():
    import uptools
    rootfile = 'data/sapta.root'

    branches = [
        b'rechit_eta',
        b'rechit_energy',
        b'rechit_x',
        b'rechit_y',
        b'rechit_z',
        b'rechit_time',
        b'rechit_flags',
        b'rechit_cluster2d',
        b'rechit_radius',
        # b'rechit_phi', b'rechit_pt',
        # b'rechit_thickness', b'rechit_layer', b'rechit_wafer_u', b'rechit_wafer_v',
        # b'rechit_cell_u', b'rechit_cell_v', b'rechit_detid', b'rechit_isHalf',
        ]

    # See also:
    # https://github.com/cms-pepr/HGCalML/blob/master/modules/datastructures/TrainData_NanoML.py#L233-L235

    def data_iterator():
        for event in uptools.iter_events(rootfile):
            r = np.sqrt(event[b'rechit_x']**2 + event[b'rechit_y']**2 + event[b'rechit_z']**2)
            theta = np.arccos(event[b'rechit_z']/r)
            X = np.stack((
                event[b'rechit_energy'],
                event[b'rechit_eta'],
                np.zeros_like(event[b'rechit_energy']),
                theta,
                event[b'rechit_radius'],
                event[b'rechit_x'],
                event[b'rechit_y'],
                event[b'rechit_z'],
                event[b'rechit_time'],
                )).T
            y = (event[b'rechit_cluster2d'] != -1).astype(np.int32)

            # Now split by endcap
            is_neg = event[b'rechit_z'] < 0
            # First pos endcap
            yield Data(
                x = torch.from_numpy(X[~is_neg]).type(torch.float),
                y = torch.from_numpy(y[~is_neg]).type(torch.int),
                batch = torch.zeros((~is_neg).sum()).long()
                )
            # Then neg endcap; be sure to flip z quantities
            X_neg = X[is_neg]
            X_neg[:,1] *= -1
            X_neg[:,8] *= -1
            yield Data(
                x = torch.from_numpy(X_neg).type(torch.float),
                y = torch.from_numpy(y[is_neg]).type(torch.int),
                batch = torch.zeros(is_neg.sum()).long()
                )

    return data_iterator

if __name__ == '__main__':
    data = single_photon_dataset()