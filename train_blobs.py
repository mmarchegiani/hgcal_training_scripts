import os, os.path as osp
from time import strftime
import tqdm
import torch
from torch_geometric.data import DataLoader
import argparse

from torch_cmspepr.dataset import BlobsDataset
from torch_cmspepr.gravnet_model import GravnetModel
import torch_cmspepr.objectcondensation as oc
# from lrscheduler import CyclicLRWithRestarts

torch.manual_seed(1009)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    args = parser.parse_args()
    if args.verbose: oc.DEBUG = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    n_epochs = 400
    batch_size = 64

    train_dataset = BlobsDataset(100 if args.dry else 10000)
    test_dataset = BlobsDataset(10 if args.dry else 2000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = GravnetModel(input_dim=2, output_dim=3).to(device)

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    loss_offset = 1. # To prevent a negative loss from ever occuring

    def loss_fn(out, data, return_components=False):
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
            return_components=return_components
            )
        if return_components:
            return out_oc
        else:
            LV, Lbeta = out_oc
            return LV + Lbeta + loss_offset

    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        pbar = tqdm.tqdm(train_loader, total=len(train_loader))
        pbar.set_postfix({'loss': '?'})
        for i, data in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            result = model(data.x, data.batch)
            loss = loss_fn(result, data)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': float(loss)})

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
        test_loss = loss_offset + loss_components['L_V']+loss_components['L_beta']
        print(f'Returning {test_loss}')
        return test_loss

    ckpt_dir = strftime('ckpts_blobs_%b%d_%H%M')
    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best: print(f'Saving epoch {checkpoint_number} as new best {"(dry)" if args.dry else ""}')
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
    pass
    main()
