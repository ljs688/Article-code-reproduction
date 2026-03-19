import argparse
import time
import torch.nn.functional as F
import torch.nn as nn
import logging
import os
import sys
import numpy as np
import random
import torch

from models import SUREfcGeneric
from Clustering import Clustering
from sure_inference import both_infer
from data_loader import loader

parser = argparse.ArgumentParser(description='SURE multi-seed evaluation')
parser.add_argument('-bs', '--batch-size', default='1024', type=int)
parser.add_argument('-e', '--epochs', default='80', type=int)
parser.add_argument('-lr', '--learn-rate', default='1e-3', type=float)
parser.add_argument('--lam', default='0.5', type=float)
parser.add_argument('-noise', '--noisy-training', type=bool, default=True)
parser.add_argument('-np', '--neg-prop', default='30', type=int)
parser.add_argument('-m', '--margin', default='5', type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('-r', '--robust', default=True, type=bool)
parser.add_argument('-t', '--switching-time', default=1.0, type=float)
parser.add_argument('-s', '--start-fine', default=False, type=bool)
parser.add_argument('--settings', default=2, type=int, help='0-PVP, 1-PSP, 2-Both')
parser.add_argument('-ap', '--aligned-prop', default='1.0', type=float)
parser.add_argument('-cp', '--complete-prop', default='1.0', type=float)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NoiseRobustLoss(nn.Module):
    def __init__(self):
        super(NoiseRobustLoss, self).__init__()

    def forward(self, pair_dist, P, margin, use_robust_loss, args):
        dist_sq = pair_dist * pair_dist
        P = P.to(torch.float32)
        N = len(P)
        if use_robust_loss == 1:
            if args.start_fine:
                loss = P * dist_sq + (1 - P) * (1 / margin) * torch.pow(
                    torch.clamp(torch.pow(pair_dist, 0.5) * (margin - pair_dist), min=0.0), 2)
            else:
                loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        else:
            loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        loss = torch.sum(loss) / (2.0 * N)
        return loss


def train(train_loader, model, criterion, optimizer, epoch, args):
    pos_dist = 0;
    neg_dist = 0;
    pos_count = 0;
    neg_count = 0
    model.train()
    for batch_idx, (x0, x1, labels, real_labels) in enumerate(train_loader):
        x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
        x0 = x0.view(x0.size()[0], -1)
        x1 = x1.view(x1.size()[0], -1)

        h0, h1, z0, z1 = model(x0, x1)
        pair_dist = F.pairwise_distance(h0, h1)

        pos_dist += torch.sum(pair_dist[labels == 1])
        neg_dist += torch.sum(pair_dist[labels == 0])
        pos_count += len(pair_dist[labels == 1])
        neg_count += len(pair_dist[labels == 0])

        ncl_loss = criterion[0](pair_dist, labels, args.margin, args.robust, args)
        ver_loss = criterion[1](x0, z0) + criterion[1](x1, z1)
        loss = ncl_loss + args.lam * ver_loss

        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pos_dist = pos_dist / (pos_count + 1e-8)
    neg_dist = neg_dist / (neg_count + 1e-8)

    if epoch != 0 and args.robust and neg_dist >= args.switching_time * args.margin and not args.start_fine:
        args.start_fine = True

    if epoch == 0 and args.margin != 1.0:
        args.margin = max(1, round((pos_dist + neg_dist).item()))

    return


def run_single_seed(dataset_name, seed):
    args.start_fine = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    train_pair_loader, all_loader, _ = loader(
        args.batch_size, args.neg_prop, args.aligned_prop,
        args.complete_prop, args.noisy_training, dataset_name
    )

    dim0, dim1 = 0, 0
    for x0, x1, *_ in train_pair_loader:
        dim0 = x0.view(x0.size(0), -1).shape[1]
        dim1 = x1.view(x1.size(0), -1).shape[1]
        break

    model = SUREfcGeneric(dim0, dim1).to(device)
    criterion_ncl = NoiseRobustLoss().to(device)
    criterion_mse = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    best_acc, best_nmi, best_ari = 0, 0, 0
    logging.getLogger().setLevel(logging.ERROR)

    for epoch in range(0, args.epochs + 1):
        if epoch == 0:
            with torch.no_grad():
                train(train_pair_loader, model, [criterion_ncl, criterion_mse], optimizer, epoch, args)
        else:
            train(train_pair_loader, model, [criterion_ncl, criterion_mse], optimizer, epoch, args)

        v0, v1, gt_label = both_infer(model, device, all_loader, args.settings)
        y_pred, ret = Clustering([v0, v1], gt_label)

        best_acc = ret['kmeans']['accuracy']
        best_nmi = ret['kmeans']['NMI']
        best_ari = ret['kmeans']['ARI']

    logging.getLogger().setLevel(logging.INFO)
    return best_acc, best_nmi, best_ari


def main():
    datasets_to_run = ['Hdigit', 'YoutubeFace']
    # datasets_to_run = ['2view-caltech101-8677sample']
    seeds_to_run = [0, 1, 2, 3, 4]

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = f"results_log_{timestamp}.txt"

    with open(log_filename, 'w', encoding='utf-8') as f:
        start_msg = "====== 开始进行多数据集&多种子评估 ======"
        print(start_msg)
        f.write(start_msg + "\n")

        for dataset in datasets_to_run:
            ds_msg = f"\n>>>> 正在测试数据集: {dataset}"
            print(ds_msg)
            f.write(ds_msg + "\n")

            acc_list, nmi_list, ari_list = [], [], []

            for seed in seeds_to_run:
                acc, nmi, ari = run_single_seed(dataset, seed)
                acc_list.append(acc)
                nmi_list.append(nmi)
                ari_list.append(ari)

                res_msg = f"  Seed {seed}: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}"
                print(res_msg)
                f.write(res_msg + "\n")

            mean_acc, std_acc = np.mean(acc_list) * 100, np.std(acc_list) * 100
            mean_nmi, std_nmi = np.mean(nmi_list) * 100, np.std(nmi_list) * 100
            mean_ari, std_ari = np.mean(ari_list) * 100, np.std(ari_list) * 100

            summary = (f"==== {dataset} 最终结果 (5次平均) ====\n"
                       f"  ACC : {mean_acc:.2f} ± {std_acc:.2f}\n"
                       f"  NMI : {mean_nmi:.2f} ± {std_nmi:.2f}\n"
                       f"  ARI : {mean_ari:.2f} ± {std_ari:.2f}\n")
            print(summary)
            f.write(summary)

    print(f"\n所有结果已保存至 {log_filename}")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    main()