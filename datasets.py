import numpy as np
import torch
from torch_geometric.data import DataLoader, Data
from torch_cmspepr.dataset import TauDataset

def tau_dataset(root='data/taus'):
    dataset = TauDataset(root)
    dataset.blacklist([ # Remove a bunch of bad events
        f'{root}/110_nanoML_98.npz',
        f'{root}/113_nanoML_13.npz',
        f'{root}/124_nanoML_77.npz',
        f'{root}/128_nanoML_70.npz',
        f'{root}/149_nanoML_90.npz',
        f'{root}/153_nanoML_22.npz',
        f'{root}/26_nanoML_93.npz',
        f'{root}/32_nanoML_45.npz',
        f'{root}/5_nanoML_51.npz',
        f'{root}/86_nanoML_97.npz',
        ])
    return dataset


def single_photon_dataset():
    import uptools
    # rootfile = 'data/sapta.root'
    rootfile = 'data/sapta_Mar24_testNanoML_CloseBy_Photon_100GeV.root'

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
            hit_e = event[b'RecHitHGC_energy']
            hit_t = event[b'RecHitHGC_time']
            hit_x = event[b'RecHitHGC_x']
            hit_y = event[b'RecHitHGC_y']
            hit_z = event[b'RecHitHGC_z']

            hit_r = np.sqrt(hit_x**2 + hit_y**2 + hit_z**2)
            hit_theta = np.arccos(hit_z/hit_r)
            hit_eta = -np.log(np.tan(hit_theta/2.))

            X = np.stack((
                hit_e,
                hit_eta,
                np.zeros_like(hit_e),
                hit_theta,
                hit_r,
                hit_x,
                hit_y,
                hit_z,
                hit_t,
                )).T
            y = (event[b'RecHitHGC_BestSimClusterIdx'] != -1).astype(np.int32)

            # Now split by endcap
            is_pos = hit_z >= 0
            # First pos endcap
            yield Data(
                x = torch.from_numpy(X[is_pos]).type(torch.float),
                y = torch.from_numpy(y[is_pos]).type(torch.int),
                batch = torch.zeros((is_pos).sum()).long()
                )
            # Then neg endcap; be sure to flip z quantities
            X_neg = X[~is_pos]
            X_neg[:,1] *= -1
            X_neg[:,8] *= -1
            yield Data(
                x = torch.from_numpy(X_neg).type(torch.float),
                y = torch.from_numpy(y[~is_pos]).type(torch.int),
                batch = torch.zeros((~is_pos).sum()).long()
                )

    return data_iterator()

if __name__ == '__main__':
    for data in single_photon_dataset():
        print(data)
        break