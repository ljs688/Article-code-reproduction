"""
Multi-seed experiment runner for GCFAggMVC
Datasets: Hdigit, Caltech-2V (2view-caltech101), MSRCV1, YouTubeFace
Seeds: 0, 1, 2, 3, 4  (5 runs each)
Metrics: ACC, NMI, ARI  ->  reported as mean±std
Results saved to ./results/<dataset>_results.txt
"""

import torch
import numpy as np
import random
import os
import argparse
from datetime import datetime

from network import GCFAggMVC
from loss import Loss
from dataloader import load_data
from metric import valid  # original valid() — we'll wrap it to capture return values


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Dataset-specific hyper-parameters (fine_tune_epochs)
DATASET_CFG = {
    "Hdigit":      {"fine_tune_epochs": 100},
    "MSRCV1":      {"fine_tune_epochs": 100},
    "Caltech101": {"fine_tune_epochs": 100},
    "YouTubeFace": {"fine_tune_epochs": 100},
}


# ─────────────────────────────────────────────────────────────────────────────
# Patched valid() that RETURNS (acc, nmi, ari) instead of only printing
# ─────────────────────────────────────────────────────────────────────────────

def valid_and_capture(model, device, dataset, view, data_size, class_num):
    """
    Calls the model in eval mode on the full dataset, runs k-means,
    and returns (acc, nmi, ari).

    This mirrors what metric.valid() typically does internally.
    If your metric.valid() already returns values, replace this body
    with a direct call and unpack the tuple.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment

    model.eval()
    all_hs = []   # per-view low-dim features
    all_labels = []

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )

    with torch.no_grad():
        for xs, y, _ in loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            _, _, hs = model(xs)
            # Use the fused representation from GCFAgg
            commonz, _ = model.GCFAgg(xs)
            all_hs.append(commonz.cpu().numpy())
            all_labels.append(y.numpy())

    feats  = np.concatenate(all_hs,    axis=0)   # (N, D)
    labels = np.concatenate(all_labels, axis=0).squeeze()

    # NaN guard — caused by gradient explosion; clipping should prevent this,
    # but replace any residual NaNs/Infs so KMeans doesn't crash
    nan_count = np.isnan(feats).sum() + np.isinf(feats).sum()
    if nan_count > 0:
        print(f"  [WARNING] Features contain {nan_count} NaN/Inf values — replacing with 0")
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    kmeans = KMeans(n_clusters=class_num, n_init=10, random_state=0)
    preds  = kmeans.fit_predict(feats)

    # ACC via Hungarian algorithm
    def cluster_acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return w[row_ind, col_ind].sum() / y_pred.size

    acc = cluster_acc(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds, average_method='arithmetic')
    ari = adjusted_rand_score(labels, preds)

    model.train()
    return acc, nmi, ari


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(dataset_name: str, seed: int, args) -> dict:
    """Train from scratch with a given seed, return {acc, nmi, ari}."""

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  Seed: {seed}")
    print(f"{'='*60}")

    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, dims, view, data_size, class_num = load_data(dataset_name)

    # 自动适配 batch_size：不能超过样本数，且保证至少有1个完整batch
    actual_batch_size = min(args.batch_size, data_size)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = GCFAggMVC(view, dims, args.low_feature_dim, args.high_feature_dim, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # Loss 的 mask 也要用实际 batch_size 初始化
    criterion = Loss(actual_batch_size, args.temperature_f, device).to(device)
    mse = torch.nn.MSELoss()

    # ── pre-training phase ────────────────────────────────────────────────
    for epoch in range(1, args.rec_epochs + 1):
        tot_loss = 0.0
        for xs, _, _ in data_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            xrs, _, _ = model(xs)
            loss = sum(mse(xs[v], xrs[v]) for v in range(view))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item()
        if epoch % 50 == 0 or epoch == 1:
            print(f'  [Pre-train] Epoch {epoch:>4d}  Loss: {tot_loss/len(data_loader):.6f}')

    # ── fine-tuning phase ─────────────────────────────────────────────────
    fine_tune_epochs = DATASET_CFG[dataset_name]["fine_tune_epochs"]
    for epoch in range(args.rec_epochs + 1, args.rec_epochs + fine_tune_epochs + 1):
        tot_loss = 0.0
        for xs, _, _ in data_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            xrs, _, hs = model(xs)
            commonz, S = model.GCFAgg(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion.Structure_guided_Contrastive_Loss(hs[v], commonz, S))
                loss_list.append(mse(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot_loss += loss.item()
        if epoch % 50 == 0 or epoch == args.rec_epochs + 1:
            print(f'  [Fine-tune] Epoch {epoch:>4d}  Loss: {tot_loss/len(data_loader):.6f}')

    # ── evaluate ──────────────────────────────────────────────────────────
    acc, nmi, ari = valid_and_capture(model, device, dataset, view, data_size, class_num)
    print(f'  >> ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}')

    # optionally save checkpoint
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), f'./models/{dataset_name}_seed{seed}.pth')

    return {"acc": acc, "nmi": nmi, "ari": ari}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate and save
# ─────────────────────────────────────────────────────────────────────────────

def fmt(mean, std, decimals=2):
    """Format as xx.xx±xx.xx (percentage)."""
    m = round(mean * 100, decimals)
    s = round(std  * 100, decimals)
    return f"{m:.{decimals}f}±{s:.{decimals}f}"


def run_dataset(dataset_name: str, seeds: list, args):
    results = []
    for seed in seeds:
        r = run_once(dataset_name, seed, args)
        results.append(r)

    accs = np.array([r["acc"] for r in results])
    nmis = np.array([r["nmi"] for r in results])
    aris = np.array([r["ari"] for r in results])

    summary = {
        "acc": (accs.mean(), accs.std()),
        "nmi": (nmis.mean(), nmis.std()),
        "ari": (aris.mean(), aris.std()),
    }

    # ── print summary ─────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  SUMMARY  —  {dataset_name}")
    print(f"{'#'*60}")
    for seed, r in zip(seeds, results):
        print(f"  Seed {seed}: ACC={r['acc']*100:.2f}  NMI={r['nmi']*100:.2f}  ARI={r['ari']*100:.2f}")
    print(f"  {'─'*40}")
    print(f"  ACC: {fmt(*summary['acc'])}  NMI: {fmt(*summary['nmi'])}  ARI: {fmt(*summary['ari'])}")
    print(f"{'#'*60}\n")

    # ── save to txt ───────────────────────────────────────────────────────
    os.makedirs('./results', exist_ok=True)
    out_path = f'./results/{dataset_name}_results.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Experiment Results — {dataset_name}\n")
        f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-seed results:\n")
        for seed, r in zip(seeds, results):
            f.write(f"  Seed {seed}: ACC={r['acc']*100:.2f}  NMI={r['nmi']*100:.2f}  ARI={r['ari']*100:.2f}\n")
        f.write("\n" + "─" * 40 + "\n")
        f.write("Mean ± Std (%):\n")
        f.write(f"  ACC : {fmt(*summary['acc'])}\n")
        f.write(f"  NMI : {fmt(*summary['nmi'])}\n")
        f.write(f"  ARI : {fmt(*summary['ari'])}\n")

    print(f"  Results saved to {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Multi-seed experiment runner')
    parser.add_argument('--datasets', nargs='+',
                        default=['Hdigit', 'Caltech101', 'MSRCV1', 'YouTubeFace'],
                        help='List of datasets to evaluate')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[0, 1, 2, 3, 4],
                        help='List of random seeds')
    parser.add_argument('--batch_size',       default=256,    type=int)
    parser.add_argument('--temperature_f',    default=0.5,    type=float)
    parser.add_argument('--learning_rate',    default=0.0003, type=float)
    parser.add_argument('--weight_decay',     default=0.,     type=float)
    parser.add_argument('--rec_epochs',       default=200,    type=int)
    parser.add_argument('--low_feature_dim',  default=512,    type=int)
    parser.add_argument('--high_feature_dim', default=128,    type=int)
    args = parser.parse_args()

    all_summaries = {}
    for ds in args.datasets:
        summary = run_dataset(ds, args.seeds, args)
        all_summaries[ds] = summary

    # ── overall comparison table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Dataset':<20}  {'ACC':>15}  {'NMI':>15}  {'ARI':>15}")
    print("=" * 70)
    for ds, s in all_summaries.items():
        print(f"  {ds:<20}  {fmt(*s['acc']):>15}  {fmt(*s['nmi']):>15}  {fmt(*s['ari']):>15}")
    print("=" * 70)

    # also save combined table
    os.makedirs('./results', exist_ok=True)
    with open('./results/all_results_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Combined Results Summary\n")
        f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Dataset':<20}  {'ACC':>15}  {'NMI':>15}  {'ARI':>15}\n")
        f.write("=" * 70 + "\n")
        for ds, s in all_summaries.items():
            f.write(f"{ds:<20}  {fmt(*s['acc']):>15}  {fmt(*s['nmi']):>15}  {fmt(*s['ari']):>15}\n")
        f.write("=" * 70 + "\n")
    print("\n  Combined summary saved to ./results/all_results_summary.txt")


if __name__ == '__main__':
    main()
