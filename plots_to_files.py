import torch
from torch_geometric.data import DataLoader
import numpy as np
import tqdm

from torch_cmspepr.dataset import TauDataset
from torch_cmspepr.gravnet_model import GravnetModel
import torch_cmspepr.objectcondensation as oc
import plotting

def pred_plot(event, out, tbeta=.1, td=1.):
    betas = torch.sigmoid(out[:,0]).numpy()
    cluster_space_coords = out[:,1:].numpy()
    clustering = oc.get_clustering_np(betas, cluster_space_coords, tbeta=tbeta, td=td)
    return plotting.get_plotly_pred(event, clustering)

def truth_plot(event):
    return plotting.get_plotly_truth(event)

def pred_clusterspace_plot(event, betas, cluster_space_coords, tbeta=.1, td=1.):
    clustering = oc.get_clustering_np(betas, cluster_space_coords, tbeta=tbeta, td=td)
    return plotting.get_plotly_clusterspace(event, cluster_space_coords, clustering)

def pca_down(cluster_space_coords: np.array, n_components: int = 3):
    from sklearn.decomposition import PCA
    dim = cluster_space_coords.shape[1]
    if dim <= n_components: return cluster_space_coords
    pca = PCA(n_components)
    out = pca.fit_transform(cluster_space_coords)
    assert out.shape == (cluster_space_coords.shape[0], n_components)
    return out

def reduced_noise_dataset():
    dataset = TauDataset('data/taus')
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
    print('Throwing away 95% of noise (good for testing ideas, not for final results)')
    dataset.reduce_noise = .95
    _, test_dataset = dataset.split(.8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader


def main():
    # _, test_dataset = TauDataset('data/taus').split(.8)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_loader = reduced_noise_dataset()

    # model = GravnetModel(input_dim=9, output_dim=4)
    # ckpt = 'ckpt_train_taus_Sep01_045842_best_6.pth.tar'

    model = GravnetModel(input_dim=9, output_dim=6, k=50)
    ckpt = 'ckpt_train_taus_Oct08_175851_best_300.pth.tar'

    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])

    tbeta = .2
    td = .5
    nmax = 20

    desc_str = f'tbeta{tbeta:.1f}_td{td:.1f}'.replace('.', 'p')

    with torch.no_grad():
        model.eval()
        for i, data in tqdm.tqdm(enumerate(test_loader), total=nmax):
            if i == nmax: break
            out = model(data.x, data.batch)
            outfile = f'plots_%b%d_{desc_str}/{i:03d}.html'
            plotting.write_html(
                outfile,
                plotting.side_by_side_html(
                    pred_plot(data, out, tbeta, td),
                    plotting.get_plotly_truth(data)
                    )
                )
            betas = torch.sigmoid(out[:,0]).numpy()
            cluster_space_coords = out[:,1:].numpy()
            clustering = oc.get_clustering_np(betas, cluster_space_coords, tbeta=tbeta, td=td)
            reduced_cluster_space_coords = pca_down(cluster_space_coords)
            plotting.write_html(
                outfile,
                plotting.side_by_side_html(
                    plotting.get_plotly_clusterspace(data, reduced_cluster_space_coords, clustering),
                    plotting.get_plotly_clusterspace(data, reduced_cluster_space_coords)
                    ),
                mode = 'a'
                )
            

if __name__ == "__main__":
    main()