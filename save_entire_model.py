import torch
import numpy as np
import glob

def get_model():
    from torch_cmspepr.gravnet_model import GravnetModelWithNoiseFilter
    model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)
    ckpt = 'ckpt_train_taus_integrated_noise_Oct20_212115_best_120.pth.tar'
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
    return model

def save_entire_model():
    torch.save(get_model(), 'gravnetwithnoisefilter.model')

def save_entire_model_jittable():
    torch.jit.save(get_model(), 'gravnetwithnoisefilter.jit')

def make_test_events():
    from torch_geometric.data import DataLoader
    from torch_cmspepr.dataset import TauDataset
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
    _, test_dataset = dataset.split(.8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    nmax = 10

    with torch.no_grad():
        model = get_model()
        model.eval()
        for i, data in enumerate(test_loader):
            if i == nmax: break
            scores_noise_filter, pass_noise_filter, out_gravnet = model(data.x, data.batch)
            np.savez(
                f'test_event_{i}.npz',
                x = data.x.numpy(),
                y = data.y.numpy(),
                batch = data.batch.numpy(),
                scores_noise_filter = scores_noise_filter.numpy(),
                pass_noise_filter = pass_noise_filter.numpy(),
                out_gravnet = out_gravnet.numpy()
                )

def test_saved_model():
    with torch.no_grad():
        model = torch.load('gravnetwithnoisefilter.model')
        model.eval()
        for npz in sorted(glob.glob('test_event_*.npz')):
            d = np.load(npz)
            x = torch.Tensor(d['x'])
            batch = torch.LongTensor(d['batch'])
            scores_noise_filter, pass_noise_filter, out_gravnet = model(x, batch)
            np.testing.assert_almost_equal(d['scores_noise_filter'], scores_noise_filter.numpy())
            np.testing.assert_almost_equal(d['pass_noise_filter'], pass_noise_filter.numpy())
            np.testing.assert_almost_equal(d['out_gravnet'], out_gravnet.numpy())


if __name__ == '__main__':
    # save_entire_model()
    # make_test_events()
    # test_saved_model()
    save_entire_model_jittable()
