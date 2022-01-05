from numpy.core.records import array
import torch
from torch.functional import _return_inverse
from torch_geometric.data import DataLoader
import numpy as np
import tqdm
from time import strftime
import os, os.path as osp

from datasets import tau_dataset
from torch_cmspepr.gravnet_model import GravnetModelWithNoiseFilter
import torch_cmspepr.objectcondensation as oc
import plotting

import plotly.graph_objects as go

from matching import match, group_matching
from colorwheel import ColorWheel

import warnings
warnings.filterwarnings("ignore")

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

def reduced_noise_testloader(reduce_noise=.95):
    dataset = tau_dataset()
    print('Throwing away 95% of noise (good for testing ideas, not for final results)')
    dataset.reduce_noise = reduce_noise
    _, test_dataset = dataset.split(.8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader


def zipped():
    import gzip, pickle
    _, test_dataset = tau_dataset().split(.8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)
    ckpt = 'ckpt_train_taus_integrated_noise_Oct20_212115_best_120.pth.tar'
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
    nmax = 1000

    with torch.no_grad():
        model.eval()
        out = []
        for i, data in tqdm.tqdm(enumerate(test_loader), total=nmax):
            if i == nmax: break
            _, pass_noise_filter, out_gravnet = model(data.x, data.batch)            
            pred_betas = torch.sigmoid(out_gravnet[:,0]).numpy()
            pred_cluster_space_coords = out_gravnet[:,1:].numpy()

            n_hits_all = data.x.size(0)
            n_hits = int(pass_noise_filter.sum())
            cluster_space_dim = pred_cluster_space_coords.shape[-1]

            x_all = data.x.numpy()
            # x = data.x[pass_noise_filter].numpy()
            y = data.y[pass_noise_filter].numpy()

            # Build back an indexed array
            cluster_ids, indices = np.unique(y, return_index=True)
            y_indexed = -1*np.ones_like(y)
            for id, index in zip(*np.unique(y, return_index=True)):
                if id==0: continue
                y_indexed[y==id] = index

            # Make it n_hits_all long again, along with the noise-filtered hits
            y_all = -1*np.ones(n_hits_all)
            y_all[pass_noise_filter] = y_indexed

            rechit_data = dict(
                recHitEnergy = x_all[:,0],
                recHitX = x_all[:,5],
                recHitY = x_all[:,6],
                recHitZ = x_all[:,7],
                )
            
            void = -1*np.ones(n_hits)
            truth_data = dict(
                truthHitAssignementIdx = y_all,
                # truthHitAssignedEnergies = void,
                # truthHitAssignedX = void,
                # truthHitAssignedY = void,
                # truthHitAssignedZ = void,
                # truthHitAssignedEta = void,
                # truthHitAssignedPhi = void,
                # truthHitAssignedT = void,
                # truthHitAssignedPIDs = void,
                # truthHitAssignedDepEnergies = void,
                )

            pred_betas_filled = np.zeros(n_hits_all)
            pred_betas_filled[pass_noise_filter] = pred_betas
            
            pred_ccoords_filled = -1*np.ones((n_hits_all, cluster_space_dim))
            pred_ccoords_filled[pass_noise_filter] = pred_cluster_space_coords

            pred_data = dict(
                pred_isnoise = pass_noise_filter.numpy(),
                pred_beta = pred_betas_filled,
                pred_ccoords = pred_ccoords_filled,
                # pred_energy = void,
                # pred_pos = -1*np.ones((n_hits, 3)),
                # pred_time = void,
                # pred_id = void,
                )

            def expand_dims(d):
                return {k : v if 'ccoords' in k else np.expand_dims(v, -1) for k, v in d.items()}

            rechit_data = expand_dims(rechit_data)
            truth_data = expand_dims(truth_data)
            pred_data = expand_dims(pred_data)

            out.append([rechit_data, truth_data, pred_data])

    outfile = strftime(f'inferences_%b%d_n{nmax}.bin.gz')
    print(f'Dumping {nmax} events to {outfile}')
    with gzip.open(outfile, 'wb') as f:
        pickle.dump(out, f)


def plots():
    _, test_dataset = tau_dataset().split(.8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)
    ckpt = 'ckpt_train_taus_integrated_noise_Oct20_212115_best_397.pth.tar'
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])

    tbeta = .2
    td = .5
    # nmax = 60
    nmax = 5

    desc_str = f'tbeta{tbeta:.1f}_td{td:.1f}'.replace('.', 'p')

    with torch.no_grad():
        model.eval()
        for i, data in tqdm.tqdm(enumerate(test_loader), total=nmax):
            if i == nmax: break
            # if i != 49: continue
            _, pass_noise_filter, out_gravnet = model(data.x, data.batch)
            pred_betas = torch.sigmoid(out_gravnet[:,0]).numpy()
            pred_cluster_space_coords = out_gravnet[:,1:].numpy()
            clustering = oc.get_clustering_np(pred_betas, pred_cluster_space_coords, tbeta=tbeta, td=td)+1
            x = data.x[pass_noise_filter]

            y = data.y.long().numpy()
            clustering_withnoise = np.zeros_like(y)
            # Insert GravNet predicted clustering
            clustering_withnoise[np.flatnonzero(pass_noise_filter)] = clustering

            colorwheel = ColorWheel()
            colorwheel.assign(0, '#bfbfbf')
            colorwheel.assign(-1, '#bfbfbf')

            # Get matches
            matches = match(y, clustering_withnoise, weights=data.x[:,0].numpy())
            for id_truth, id_pred, iom in zip(*matches):
                id_truth = int(id_truth)
                id_pred = int(id_pred)
                # print(f'{id_truth=}, {id_pred=}, {iom=}')
                if id_truth in colorwheel.assigned_colors and id_pred in colorwheel.assigned_colors:
                    continue
                elif id_truth in colorwheel.assigned_colors:
                    colorwheel.assign(id_pred, colorwheel(id_truth))
                elif id_pred in colorwheel.assigned_colors:
                    colorwheel.assign(id_truth, colorwheel(id_pred))
                else:
                    colorwheel.assign(id_pred, colorwheel(id_truth))

            outfile = strftime(f'plots_%b%d_{desc_str}/{i:03d}.html')
            
            directory = osp.dirname(outfile)
            if directory and not osp.isdir(directory): os.makedirs(directory)

            with open(outfile, 'w') as f:
                f.write(
                    '<div style="display:flex">'
                    '  <div style="flex:50%">'
                    '    <h2>Predicted clustering</h2>'
                    # '    <p>Cluster -2 is predicted noise by the noise filter, -1 is predicted noise by the GravNet model</p>'
                    '    </div>'
                    '  <div style="flex:50%">'
                    '    <h2>Truth clustering</h2>'
                    '    </div>'
                    '</div>'
                    )

            plotting.write_html(
                outfile,
                plotting.side_by_side_html(
                    plotting.get_plotly_pred(data.x, clustering_withnoise, colorwheel=colorwheel),
                    plotting.get_plotly_truth(data, colorwheel=colorwheel)
                    ),
                mode = 'a'
                )

            with open(outfile, 'a') as f:
                f.write(
                    '<div style="display:flex">'
                    '  <div style="flex:50%">'
                    '    <h2>Predicted clustering in clustering space</h2>'
                    '    </div>'
                    '  <div style="flex:50%">'
                    '    <h2>Truth clustering in clustering space</h2>'
                    '    </div>'
                    '</div>'
                    )

            y = data.y.long()[pass_noise_filter].numpy()
            reduced_cluster_space_coords = pca_down(pred_cluster_space_coords)
            plotting.write_html(
                outfile,
                plotting.side_by_side_html(
                    plotting.get_plotly_clusterspace_xy(x, reduced_cluster_space_coords, clustering, colorwheel=colorwheel),
                    plotting.get_plotly_clusterspace_xy(x, reduced_cluster_space_coords, y, colorwheel=colorwheel),
                    ),
                mode = 'a'
                )

def stats():
    _, test_dataset = tau_dataset().split(.8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)
    ckpt = 'ckpt_train_taus_integrated_noise_Oct20_212115_best_397.pth.tar'
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
    tbeta = .2
    td = .5
    nmax = 200

    nhits_pred = []
    esum_pred = []
    nhits_truth = []
    esum_truth = []
    pdgid_truth = []
    is_strict_em_truth = []

    with torch.no_grad():
        model.eval()
        for i, data in tqdm.tqdm(enumerate(test_loader), total=nmax):
            if i == nmax: break
            _, pass_noise_filter, out_gravnet = model(data.x, data.batch)
            pred_betas = torch.sigmoid(out_gravnet[:,0]).numpy()
            pred_cluster_space_coords = out_gravnet[:,1:].numpy()
            clustering = oc.get_clustering_np(pred_betas, pred_cluster_space_coords, tbeta=tbeta, td=td)+1
            x = data.x[pass_noise_filter]
            y = data.y.long().numpy()
            clustering_withnoise = np.zeros_like(y)
            # Insert GravNet predicted clustering
            clustering_withnoise[np.flatnonzero(pass_noise_filter)] = clustering
            energies = data.x[:,0].numpy()
            truth_pdgids = data.truth_cluster_props[:,-1]

            i1s, i2s, _ = match(y, clustering_withnoise, weights=data.x[:,0].numpy())
            matches = group_matching(i1s, i2s)

            for truth_ids, pred_ids in matches:
                truth_cluster_pdgids = []
                sel_truth_hits = np.zeros_like(y, dtype=bool)
                for truth_id in truth_ids:
                    sel_truth_hits[y==truth_id] = True
                    truth_cluster_pdgids.append(truth_pdgids[y==truth_id][0])
                truth_cluster_pdgids = np.array(truth_cluster_pdgids)

                sel_pred_hits = np.zeros_like(y, dtype=bool)
                for pred_id in pred_ids: sel_pred_hits[clustering_withnoise==pred_id] = True

                most_likely_pdgid = truth_cluster_pdgids[0]
                if np.any(truth_cluster_pdgids != most_likely_pdgid):
                    print(f'Warning: Treating match {truth_ids}/{pred_ids} as {most_likely_pdgid}, but all truth pdgids are: {truth_cluster_pdgids}')

                nhits_pred.append(sel_pred_hits.sum())
                esum_pred.append(energies[sel_pred_hits].sum())
                nhits_truth.append(sel_truth_hits.sum())
                esum_truth.append(energies[sel_truth_hits].sum())
                pdgid_truth.append(most_likely_pdgid)
                is_strict_em_truth.append(np.all((np.abs(truth_cluster_pdgids) == 11) | (np.abs(truth_cluster_pdgids) == 22)))

    np.savez(
        'histogram_input.npz',
        nhits_pred = np.array(nhits_pred),
        esum_pred = np.array(esum_pred),
        nhits_truth = np.array(nhits_truth),
        esum_truth = np.array(esum_truth),
        pdgid_truth = np.array(pdgid_truth),
        is_strict_em_truth = np.array(is_strict_em_truth)
        )
    

if __name__ == "__main__":
    # stats()
    plots()
    # zipped()