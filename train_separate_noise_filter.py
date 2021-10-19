import os, os.path as osp
from time import strftime
import tqdm
import torch
from torch_geometric.data import DataLoader
import argparse
import numpy as np

from torch_cmspepr.dataset import TauDataset
from torch_cmspepr.gravnet_model import GravnetModel
import torch_cmspepr.objectcondensation as oc
from datasets import tau_dataset
from lrscheduler import CyclicLRWithRestarts

torch.manual_seed(1009)




import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_min, scatter_max, scatter_mean

from torch_cmspepr import GravNetConv
from torch_cmspepr.objectcondensation import scatter_count

from typing import Union, List


class NoiseFilterModel(nn.Module):

    def __init__(
        self, 
        input_dim: int=5,
        output_dim: int=2,
        ):
        super(NoiseFilterModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.ReLU(),
            nn.Softmax()
            )

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        out = self.network(x)
        assert out.size() == (x.size(0), 2)
        return out


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
    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")


    loss = nn.BCELoss()
    def loss_fn(out, data):
        is_signal = (data.y != 0).float()
        loss_value = loss(out[:,0], is_signal)
        return loss_value


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
        # avg_acc = 0
        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                result = model(data.x)
                loss_value = loss_fn(result, data)
                avg_loss += loss_value
        avg_loss /= n_batches
        print(f'{avg_loss=}')
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

def debug():
    oc.DEBUG = True
    dataset = TauDataset('data/taus')
    dataset.npzs = [
        # 'data/taus/49_nanoML_84.npz',
        # 'data/taus/37_nanoML_4.npz',
        'data/taus/26_nanoML_93.npz',
        # 'data/taus/142_nanoML_75.npz',
        ]
    for data in DataLoader(dataset, batch_size=len(dataset), shuffle=False): break
    print(data.y.sum())
    model = GravnetModel(input_dim=9, output_dim=4)
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.batch)
    pred_edc = torch.sigmoid(out[:,0])
    pred_cluster_space_coords = out[:,1:4]
    out_oc = oc.calc_LV_Lbeta(
        pred_edc,
        pred_cluster_space_coords,
        data.y.long(),
        data.batch.long()
        )

def run_profile():
    from torch.profiler import profile, record_function, ProfilerActivity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    batch_size = 2
    n_batches = 2
    shuffle = True
    dataset = TauDataset('data/taus')
    dataset.npzs = dataset.npzs[:batch_size*n_batches]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f'Running profiling for {len(dataset)} events, batch_size={batch_size}, {len(loader)} batches')

    model = GravnetModel(input_dim=9, output_dim=8).to(device)
    epoch_size = len(loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)

    print('Start limited training loop')
    model.train()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            pbar = tqdm.tqdm(loader, total=len(loader))
            pbar.set_postfix({'loss': '?'})
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                loss = loss_fn(result, data)
                print(f'loss={float(loss)}')
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': float(loss)})
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=10))
    # Other valid keys:
    # cpu_time, cuda_time, cpu_time_total, cuda_time_total, cpu_memory_usage,
    # cuda_memory_usage, self_cpu_memory_usage, self_cuda_memory_usage, count

if __name__ == '__main__':
    pass
    main()
    # debug()
    # run_profile()