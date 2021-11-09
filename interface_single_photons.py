import torch
import numpy as np
import glob
import uproot
# from torch_cmspepr.dataset import incremental_cluster_index_np


def interface():
    t = uproot.open('testNanoML_Diphoton_test.root')['Events']
    arrays = t.arrays()

    with torch.no_grad():
        model = torch.load('gravnetwithnoisefilter.model')
        model.eval()

        for i in range(t.num_entries):
            if i == 2: break
            e = np.array(arrays['RecHitHGC_energy'][i])
            x = np.array(arrays['RecHitHGC_x'][i])
            y = np.array(arrays['RecHitHGC_y'][i])
            z = np.array(arrays['RecHitHGC_z'][i])
            t = np.array(arrays['RecHitHGC_time'][i])

            nhits = e.shape[0]

            r = np.sqrt(x**2 + y**2 + z**2)

            d = np.sqrt(x**2 + y**2)
            theta = np.arctan2(d, z)
            eta = -np.log(np.tan(theta/2.))

            # Make sure phi is within -pi..pi
            phi = np.arctan2(x, y) % (2.*np.pi)
            phi[phi > np.pi] -= 2.*np.pi

            features = np.vstack((
                e,
                eta,
                np.zeros_like(e),
                theta,
                r,
                x,
                y,
                z,
                t
                )).T

            assert features.shape == (nhits, 9)

            # This should be the truth clustering
            y = np.array(arrays['RecHitHGC_BestMergedSimClusterIdx'][i])
            # y = incremental_cluster_index_np(y, noise_index=-1)
            assert y.shape == (nhits,)


            X = torch.Tensor(features)
            batch = torch.zeros(nhits, dtype=torch.long)
            scores_noise_filter, pass_noise_filter, out_gravnet = model(X, batch)

            print(scores_noise_filter, pass_noise_filter, out_gravnet)


if __name__ == '__main__':
    interface()