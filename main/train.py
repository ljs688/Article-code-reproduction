import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import list_datasets
from main.experiment import run_dataset_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run GHICMC on a single dataset.")
    parser.add_argument(
        "--dataset",
        default="Hdigit",
        choices=list_datasets(),
        help="Dataset to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for both mask generation and model initialization.",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.5,
        help="Missing rate for the incomplete multi-view setting.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for debugging or smoke tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dataset_experiment(
        dataset=args.dataset,
        seeds=[args.seed],
        missing_rate=args.missing_rate,
        epochs_override=args.epochs,
    )
