import os, os.path as osp
from time import strftime
import tqdm
import torch
from torch_geometric.data import DataLoader
import argparse

from torch_cmspepr.dataset import TauDataset
from torch_cmspepr.gravnet_model import GravnetModel
import torch_cmspepr.objectcondensation as oc


torch.manual_seed(1009)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    args = parser.parse_args()
    if args.verbose: oc.DEBUG = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    n_epochs = 200
    batch_size = 30

    shuffle = True
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
    dataset.reduce_noise = .90
    print(f'Throwing away {dataset.reduce_noise*100:.0f}% of noise at the truth level')
    if args.dry:
        keep = .005
        print(f'Keeping only {100.*keep:.1f}% of events for debugging')
        dataset, _ = dataset.split(keep)

    train_dataset, test_dataset = dataset.split(.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = GravnetModel(input_dim=9, output_dim=3, k=5, n_gravnet_blocks=2, n_postgn_dense_blocks=2).to(device)

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


    def loss_fn(out, data, i_epoch=None, return_components=False):
        device = out.device
        pred_betas = torch.sigmoid(out[:,0])
        pred_cluster_space_coords = out[:,1:]
        assert all(t.device == device for t in [
            pred_betas, pred_cluster_space_coords, data.y, data.batch,
            ])
        out_oc = oc.calc_LV_Lbeta(
            pred_betas,
            pred_cluster_space_coords,
            data.y.long(),
            data.batch,
            return_components=return_components,
            beta_term_option='short-range-potential',
            )
        if return_components:
            return out_oc
        else:
            LV, Lbeta = out_oc
            if i_epoch <= 7:
                return LV + 1. # Add 1. to prevent a negative loss from occuring
            else:
                return LV + Lbeta + 1.

    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({'loss': '?'})
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                loss = loss_fn(result, data, i_epoch=epoch)
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': float(loss)})
                # if i == 2: raise Exception
        except Exception:
            print('Exception encountered:', data, ', npzs:')
            print('  ' + '\n  '.join([train_dataset.npzs[int(i)] for i in data.inpz]))
            raise

    def test(epoch):
        N_test = len(test_loader)
        loss_components = {}
        def update(components):
            for key, value in components.items():
                if not key in loss_components: loss_components[key] = 0.
                loss_components[key] += value
        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model(data.x, data.batch)
                update(loss_fn(result, data, return_components=True))
        # Divide by number of entries
        for key in loss_components:
            loss_components[key] /= N_test
        # Compute total loss and do printout
        print('test ' + oc.formatted_loss_components_string(loss_components))
        test_loss = 1. + loss_components['L_V']+loss_components['L_beta']
        print(f'Returning {test_loss}')
        return test_loss

    ckpt_dir = strftime('ckpts_gravnet_benchmark_%b%d_%H%M')
    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        if not args.dry:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(model=model.state_dict()), ckpt)

    min_loss = 1e9
    for i_epoch in range(n_epochs):
        train(i_epoch)
        write_checkpoint(i_epoch)
        test_loss = test(i_epoch)
        if test_loss < min_loss:
            min_loss = test_loss
            write_checkpoint(i_epoch, best=True)

if __name__ == '__main__':
    main()
    # debug()
    # run_profile()