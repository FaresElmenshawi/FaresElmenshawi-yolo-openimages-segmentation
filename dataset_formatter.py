import argparse
import os

from dataset_utils.format_files import format_directory, join_if_not_none
from dataset_utils.format_dataset import create_dataset
from dataset_utils.create_classes import extract_classes


def main(classes, classes_csv='train/metadata/classes.csv', working_dir=None, train_img_src='train/data', train_label_src='train/labels/masks',
         train_img_dist='images/train', train_label_dist='labels/train', val_img_src=None, val_label_src=None, val_img_dist=None, val_label_dist=None, val_split=0.1,
         ignore_warning=False):
    """Main function to format open images dataset for yolo training for segmentation.

    Args:
        classes (list): List of classes.
        classes_csv (str): Path to classes CSV file. Defaults to 'train/metadata/classes.csv' (relative to working directory).
        working_dir (str): Working directory. Defaults to None.
        train_img_src (str): Path to training image source directory. Defaults to 'train/data' (relative to working directory).
        train_label_src (str): Path to training label source directory. Defaults to 'train/labels/masks' (relative to working directory).
        train_img_dist (str): Path to training image destination directory. Defaults to 'images/train' (relative to working directory).
        train_label_dist (str): Path to training label destination directory. Defaults to 'labels/train' (relative to working directory).
        val_img_src (str): Path to validation image source directory. Defaults to None.
        val_label_src (str): Path to validation label source directory. Defaults to None.
        val_img_dist (str): Path to validation image destination directory. Defaults to None.
        val_label_dist (str): Path to validation label destination directory. Defaults to None.
        val_split (float): Validation split ratio. Defaults to 0.1.
        ignore_warning (bool): Whether to ignore warnings or not. Defaults to False.
    """
    if working_dir is None:
        working_dir = os.path.dirname(os.getcwd())
        if not ignore_warning:
            user_input = input(f"Warning: working dir is not passed using parent of the current directory as the working directory: {working_dir}\n"
                               f"Do you want to continue? (Y/n)\n")
            if user_input not in ['y', 'Y']:
                return

    if val_img_src is None or val_label_src is None:
        if not ignore_warning:
            user_input = input(
                f"Warning: No validation source directories are passed. This assumes you have no validation data, "
                f"{val_split} of the train data will be used for validation.\n"
                f"If you want to confirm, press Y; otherwise, these paths will be used for the validation data:\n"
                f"val_img_src: {train_img_src.replace('train', 'validation')} (relative to working directory)\n"
                f"val_label_src: {train_label_src.replace('train', 'validation')} (relative to working directory) (Y/n)\n")

            if user_input not in ['y', 'Y']:
                val_img_src = train_img_src.replace('train', 'validation')
                val_label_src = train_label_src.replace('train', 'validation')

    classes_csv, train_img_src, train_label_src, train_img_dist, train_label_dist, val_img_src, val_label_src, val_img_dist, val_label_dist = \
        join_if_not_none(working_dir, classes_csv, train_img_src, train_label_src, train_img_dist, train_label_dist, val_img_src, val_label_src, val_img_dist, val_label_dist)

    code_list, code_to_info = extract_classes(classes, os.path.join(working_dir, classes_csv))

    format_directory(working_dir, classes, split_names=("train", "val"))

    create_dataset(code_list, train_img_src, train_label_src, train_img_dist, train_label_dist, code_to_info,
                   val_img_src=val_img_src, val_label_src=val_label_src, val_img_dist=val_img_dist,
                   val_label_dist=val_label_dist,
                   val_split=val_split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to format open images dataset for yolo training for segmentation')

    parser.add_argument("--classes", nargs='+', required=True, help="List of classes")
    parser.add_argument('--working-dir', default=None, help='Working directory')
    parser.add_argument('--classes-csv', default='train/metadata/classes.csv', help='Path to classes CSV file (relative to working directory)')
    parser.add_argument('--train-img-src', default='train/data', help='Path to training image source directory (relative to working directory)')
    parser.add_argument('--train-label-src', default='train/labels/masks', help='Path to training label source directory (relative to working directory)')
    parser.add_argument('--train-img-dist', default='images/train', help='Path to training image destination directory (relative to working directory)')
    parser.add_argument('--train-label-dist', default='labels/train', help='Path to training label destination directory (relative to working directory)')
    parser.add_argument('--val-img-src', default=None, help='Path to validation image source directory (relative to working directory)')
    parser.add_argument('--val-label-src', default=None, help='Path to validation label source directory (relative to working directory)')
    parser.add_argument('--val-img-dist', default=None, help='Path to validation image destination directory (relative to working directory)')
    parser.add_argument('--val-label-dist', default=None, help='Path to validation label destination directory (relative to working directory)')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--ignore_warnings', action='store_true', help='Enable the new feature')

    args = parser.parse_args()

    main(args.classes, args.classes_csv, args.working_dir, args.train_img_src, args.train_label_src,
         args.train_img_dist, args.train_label_dist, args.val_img_src, args.val_label_src, args.val_img_dist,
         args.val_label_dist, args.val_split, args.ignore_warnings)
