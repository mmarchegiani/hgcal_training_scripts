import os, os.path as osp
from time import strftime
import tqdm
import argparse
import numpy as np

import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from torch_cmspepr.gravnet_model import GravnetModelWithNoiseFilter
import torch_cmspepr.objectcondensation as oc

from datasets import tau_dataset
from lrscheduler import CyclicLRWithRestarts

from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1009)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    parser.add_argument('--ckptdir', type=str)
    parser.add_argument('--root', type=str, default='data/taus', help='Root directory for tau dataset')
    args = parser.parse_args()
    if args.verbose: oc.DEBUG = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    n_epochs = 400
    batch_size = 4

    shuffle = True
    dataset = tau_dataset(args.root)
    if args.dry:
        keep = .005
        # keep = .2
        print(f'Keeping only {100.*keep:.1f}% of events for debugging')
        dataset, _ = dataset.split(keep)
    train_dataset, test_dataset = dataset.split(.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = GravnetModelWithNoiseFilter(
        input_dim=9, output_dim=6, k=50, signal_threshold=.05
        ).to(device)

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    logs_dir = strftime('logs_gravnet_%b%d_%H%M') if args.ckptdir is None else os.path.join(args.ckptdir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logs_dir)

    n_noise_filter_epochs = 10
    n_only_clustering_epochs = 10
    loss_offset = 1. # To prevent a negative loss from ever occuring

    def only_noise_loss(result, data):
        return F.nll_loss(result, (data.y != 0).long())

    def loss_fn(out, data, i_epoch=None, return_components=False):
        out_noise_filter, pass_noise_filter, out_gravnet = out
        device = out_gravnet.device
        pred_betas = torch.sigmoid(out_gravnet[:,0])
        pred_cluster_space_coords = out_gravnet[:,1:]
        assert all(t.device == device for t in [
            pred_betas, pred_cluster_space_coords, data.y, data.batch,
            ])
        y = data.y.long()[pass_noise_filter]
        batch = data.batch[pass_noise_filter]
        y = oc.reincrementalize(y, batch)
        if torch.all(y == 0):
            print('Problem: All signal hits filtered out by noise filter')
            print(f'n passing noise filter: {pass_noise_filter.sum()} of {pass_noise_filter.size(0)}')
            print(f'n signal hits: {(data.y > 0).sum()}')
            raise Exception()
        oc_loss = oc.calc_LV_Lbeta(
            pred_betas,
            pred_cluster_space_coords,
            y,
            batch,
            return_components=return_components,
            beta_term_option='short-range-potential',
            )
        L_noise_filter = only_noise_loss(out_noise_filter, data)
        if return_components:
            oc_loss['L_noise_filter'] = L_noise_filter
            return oc_loss
        else:
            LV, Lbeta = oc_loss
            if i_epoch <= n_only_clustering_epochs:
                return L_noise_filter + LV + loss_offset
            else:
                return L_noise_filter + LV + Lbeta + loss_offset

    def train(epoch):
        print('Training epoch', epoch)
        if epoch <= n_noise_filter_epochs:
            print(f'(only using L_V; L_beta is ignored until epoch {n_only_clustering_epochs+1})')
        model.train()
        scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({'loss': '?'})
            total_loss = 0.
            num_batches = len(train_loader)
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                loss = loss_fn(result, data, i_epoch=epoch)
                total_loss += float(loss)
                writer.add_scalar('Loss/train_batch', float(loss), epoch * len(train_loader) + i) # Log loss per batch to tensorboard
                loss.backward()
                optimizer.step()
                scheduler.batch_step()
                pbar.set_postfix({'loss': float(loss)})
                # if i == 2: raise Exception
            avg_loss = total_loss / num_batches
            writer.add_scalar('Loss/train', avg_loss, epoch)
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
        conf_mat = np.zeros((2,2))
        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model(data.x, data.batch)
                update(loss_fn(result, data, return_components=True))
                pass_noise_filter = result[1]
                conf_mat[0,0] += (pass_noise_filter[data.y == 0] == 0).sum()
                conf_mat[1,1] += (pass_noise_filter[data.y == 1] == 1).sum()
                conf_mat[0,1] += (pass_noise_filter[data.y == 0] == 1).sum()
                conf_mat[1,0] += (pass_noise_filter[data.y == 1] == 0).sum()
        # Divide by number of entries
        for key in loss_components:
            loss_components[key] /= N_test
        # Compute total loss and do printout
        print(f'test losses epoch {epoch}:')
        print(oc.formatted_loss_components_string(loss_components))
        test_loss = loss_offset + loss_components['L_V'] + loss_components['L_noise_filter']
        if epoch > n_only_clustering_epochs:
            test_loss += loss_components['L_beta']
        writer.add_scalar('Loss/test', test_loss, epoch)
        print(f'Returning {test_loss}')
        print('Noise filter confusion matrix:')
        print(conf_mat / np.expand_dims(conf_mat.sum(axis=1), axis=1))
        return test_loss

    ckpt_dir = strftime('ckpts_gravnet_%b%d_%H%M') if args.ckptdir is None else args.ckptdir
    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        if not args.dry:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(model=model.state_dict()), ckpt)



    only_noise_optimizer = torch.optim.AdamW(model.noise_filter.parameters(), lr=1e-3, weight_decay=1e-4)

    def train_only_noise_filter(epoch):
        print('Training epoch', epoch)
        model.train()
        pbar = tqdm.tqdm(train_loader, total=len(train_loader))
        pbar.set_postfix({'loss': '?'})
        for i, data in enumerate(pbar):
            data = data.to(device)
            only_noise_optimizer.zero_grad()
            result = model.noise_filter(data.x)
            loss = only_noise_loss(result, data)
            loss.backward()
            only_noise_optimizer.step()
            pbar.set_postfix({'loss': float(loss)})

    def test_only_noise_filter(epoch):
        n_batches = len(test_loader)
        avg_loss = 0.
        conf_mat = np.zeros((2,2))
        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model.noise_filter(data.x)
                loss_value = only_noise_loss(result, data)
                avg_loss += loss_value
                pred = result.argmax(dim=1)
                conf_mat[0,0] += (pred[data.y == 0] == 0).sum()
                conf_mat[1,1] += (pred[data.y == 1] == 1).sum()
                conf_mat[0,1] += (pred[data.y == 0] == 1).sum()
                conf_mat[1,0] += (pred[data.y == 1] == 0).sum()
        avg_loss /= n_batches
        print(f'avg test loss: {avg_loss}; conf mat:')
        print(conf_mat)
        print(conf_mat / np.expand_dims(conf_mat.sum(axis=1), axis=1))
        return avg_loss

    print(f'TRAINING ONLY THE NOISE FILTER FIRST FOR {n_noise_filter_epochs} EPOCHS')
    for i_epoch in range(n_noise_filter_epochs):
        train_only_noise_filter(i_epoch)
        test_only_noise_filter(i_epoch)
        
    print(f'NOW STARTING FULL TRAINING')
    min_loss = 1e9
    for i_epoch in range(n_epochs):
        train(i_epoch)
        write_checkpoint(i_epoch)
        test_loss = test(i_epoch)
        if test_loss < min_loss:
            min_loss = test_loss
            write_checkpoint(i_epoch, best=True)

    # Close the TensorBoard writer
    writer.close()


if __name__ == '__main__':
    main()
