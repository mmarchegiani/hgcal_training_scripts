import os, os.path as osp
from time import strftime
import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch import Tensor

from torch_cmspepr.dataset import TauDataset
from torch_cmspepr.gravnet_model import NoiseFilterModel
import torch_cmspepr.objectcondensation as oc

from datasets import tau_dataset
from lrscheduler import CyclicLRWithRestarts

torch.manual_seed(1009)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    parser.add_argument('--ckptdir', type=str)
    args = parser.parse_args()
    if args.verbose: oc.DEBUG = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    n_epochs = 400
    batch_size = 4

    dataset = tau_dataset()
    if args.dry:
        keep = .0025
        print(f'Keeping only {100.*keep:.2f}% of events for debugging')
        dataset, _ = dataset.split(keep)
        batch_size = 2
    train_dataset, test_dataset = dataset.split(.8)
    shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = NoiseFilterModel(input_dim=9, output_dim=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epoch_size = len(train_loader.dataset)
    # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    def loss_fn(out, data):
        is_signal = (data.y != 0).long()
        return F.nll_loss(out, is_signal)

    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        # scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({'loss': '?'})
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x)
                loss = loss_fn(result, data)
                loss.backward()
                optimizer.step()
                # scheduler.batch_step()
                pbar.set_postfix({'loss': float(loss)})
                # if i == 2: raise Exception
        except Exception:
            print('Exception encountered:', data, ', npzs:')
            print('  ' + '\n  '.join([train_dataset.npzs[int(i)] for i in data.inpz]))
            raise

    def test(epoch):
        n_batches = len(test_loader)
        avg_loss = 0.
        conf_mat = np.zeros((2,2))
        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model(data.x)
                loss_value = loss_fn(result, data)
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

    ckpt_dir = strftime('ckpts_separate_noise_filter_%b%d_%H%M') if args.ckptdir is None else args.ckptdir
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