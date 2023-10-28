import os


def join_if_not_none(base_dir, *paths):
    """
    Join a base directory with a list of paths, only if the path is not None.

    Args:
        base_dir (str): The base directory to join with the paths.
        *paths (str): Variable number of paths to join with the base directory.

    Returns:
        list of str: A list of joined paths. If a path is None, it remains as None.
    """
    return [os.path.join(base_dir, path) if path is not None else None for path in paths]


def create_class_dir(train_labels_path, train_images_path, val_labels_path, val_images_path, name):
    """
    Create class directories for training and validation images and labels.

    Args:
        train_labels_path (str): The path to the training labels directory.
        train_images_path (str): The path to the training images directory.
        val_labels_path (str): The path to the validation labels directory.
        val_images_path (str): The path to the validation images directory.
        name (str): The name of the class for which directories are created.
    """
    for directory in [train_labels_path, train_images_path, val_labels_path, val_images_path]:
        os.makedirs(os.path.join(directory, name), exist_ok=True)


def format_directory(main_path, label_names, split_names=("train", "val")):
    """
    Format a directory structure for storing images and labels for different classes and splits.

    Args:
        main_path (str): The main directory path where images and labels will be organized.
        label_names (list): A list of class labels.
        split_names (tuple): Names of the splits (e.g., train and validation). Default is ("train", "val").
    """
    main_images_path = os.path.join(main_path, "images")
    main_labels_path = os.path.join(main_path, "labels")

    for directory in [main_images_path, main_labels_path]:
        os.makedirs(directory, exist_ok=True)

    for split_name in split_names:
        split_images_path = os.path.join(main_images_path, split_name)
        split_labels_path = os.path.join(main_labels_path, split_name)

        for directory in [split_images_path, split_labels_path]:
            os.makedirs(directory, exist_ok=True)

        for label in label_names:
            create_class_dir(split_labels_path, split_images_path, split_labels_path, split_images_path, label)


def main(args):
    format_directory(args.main_path, args.label_names, args.split_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format a directory structure for storing images and labels.")
    parser.add_argument("main_path", type=str, help="The main directory path where images and labels will be organized.")
    parser.add_argument("label_names", nargs="+", type=str, help="A list of class labels.")
    parser.add_argument("--split_names", nargs=2, default=("train", "val"), help="Names of the splits (e.g., train and validation). Default is 'train' and 'val'.")

    args = parser.parse_args()
    main(args)
