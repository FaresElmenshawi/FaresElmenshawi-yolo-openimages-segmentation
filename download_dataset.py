import fiftyone.zoo as foz
import os
import argparse


def download_dataset(dataset_dir, max_samples, splits, classes):
    if dataset_dir is None:
        # Use the parent of the current directory as the default dataset directory
        dataset_dir = os.path.join(os.getcwd(), os.pardir)

        print("Warning: Using the parent of the current directory as the dataset directory.")

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        splits=splits,
        label_types=["segmentations"],
        classes=classes,
        max_samples=max_samples,
        shuffle=True,
        dataset_dir=dataset_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from FiftyOne Zoo")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory where the dataset will be downloaded",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to download",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        type=str,
        default=["train"],  # Default to "train" split
        help="Splits to download (e.g., train, validation)",
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        type=str,
        help="Classes to download",
    )

    args = parser.parse_args()

    download_dataset(args.dataset_dir, args.max_samples, args.splits, args.classes)
