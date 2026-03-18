import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import SKIPPED_DATASETS, list_datasets
from main.experiment import run_dataset_experiment
from utils.util import write_summary_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run GHICMC on the supported datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *list_datasets(), *SKIPPED_DATASETS.keys()],
        help="Datasets to run. Use 'all' for the default supported datasets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds used for both mask generation and model initialization.",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.5,
        help="Missing rate for the incomplete multi-view setting.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "result" / "three_datasets_5seeds.txt"),
        help="Output summary txt path.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for debugging or smoke tests.",
    )
    return parser.parse_args()


def resolve_datasets(dataset_args):
    if "all" in dataset_args:
        return list_datasets()
    return dataset_args


if __name__ == "__main__":
    args = parse_args()
    datasets = resolve_datasets(args.datasets)

    summary_rows = []
    for dataset in datasets:
        if dataset in SKIPPED_DATASETS:
            raise ValueError(f"{dataset}: {SKIPPED_DATASETS[dataset]}")
        summary_rows.append(
            run_dataset_experiment(
                dataset=dataset,
                seeds=args.seeds,
                missing_rate=args.missing_rate,
                epochs_override=args.epochs,
            )
        )

    write_summary_results(args.output, summary_rows)
    print(f"Summary written to {args.output}")
