from shutil import move, copyfile
from dataset_utils.convert_to_yolo import convert_to_yolo
import os
from tqdm import tqdm
from dataset_utils.split_data import train_test_split


def decide_search_dir(img):
    """
    Determine the search directory based on the first character of the image name.

    Args:
        img (str): The name of the image file.

    Returns:
        str: The search directory based on the first character of the image name.
    """
    if img[0].isalpha():
        return img[0].upper()
    else:
        return img[0]


def create_dataset_subset(class_names, img_main_dir, label_main_dir, image_dist_dir, label_dist_dir,
                          class_code_to_info, split_name='Train'):
    """
    Create a subset of a dataset based on class names and copy images and labels to new directories.

    Args:
        class_names (list): List of class names to include in the subset.
        img_main_dir (str): The directory containing the source images.
        label_main_dir (str): The directory containing the source labels.
        image_dist_dir (str): The directory to copy the subset images to.
        label_dist_dir (str): The directory to copy the subset labels to.
        class_code_to_info (dict): A dictionary mapping class names to class codes and other information.

    Parameters
    ----------
    split_name
    """
    img_names = os.listdir(img_main_dir)
    for img_name in tqdm(img_names, desc=f"Creating {split_name} dataset"):
        label_sub_dir = os.path.join(label_main_dir, decide_search_dir(img_name))
        for mask_name in os.listdir(label_sub_dir):
            for class_name in class_names:
                if class_name in mask_name and img_name.replace(".jpg", "") in mask_name:
                    mask_src_path = os.path.join(label_sub_dir, mask_name)
                    label_dist_path = os.path.join(
                        os.path.join(label_dist_dir, class_code_to_info[class_name][1]),
                        mask_name.split('_')[0] + ".txt")
                    convert_to_yolo(mask_src_path, label_dist_path, class_code_to_info[class_name][0])
                    img_src_path = os.path.join(img_main_dir, img_name)
                    img_dist_path = os.path.join(
                        os.path.join(image_dist_dir, class_code_to_info[class_name][1]), img_name)
                    copyfile(img_src_path, img_dist_path)
                    break


def create_dataset(class_code_list, train_img_src, train_label_src, train_img_dist, train_label_dist,
                   class_code_to_info,
                   val_img_src=None, val_label_src=None, val_img_dist=None, val_label_dist=None, val_split=0.1):
    """
    Create a dataset with training and validation subsets based on class codes and split it into two directories.

    Args:
        class_code_list (list): List of class codes to include in the dataset.
        train_img_src (str): Directory containing the source training images.
        train_label_src (str): Directory containing the source training labels.
        train_img_dist (str): Directory to copy the training images to.
        train_label_dist (str): Directory to copy the training labels to.
        class_code_to_info (dict): A dictionary mapping class names to class codes and other information.
        val_img_src (str, optional): Directory containing the source validation images (default is None).
        val_label_src (str, optional): Directory containing the source validation labels (default is None).
        val_img_dist (str, optional): Directory to copy the validation images (default is None).
        val_label_dist (str, optional): Directory to copy the validation labels (default is None).
        val_split (float, optional): The proportion of data to be used for validation (default is 0.1).
    """
    create_dataset_subset(class_code_list, train_img_src, train_label_src, train_img_dist, train_label_dist,
                          class_code_to_info, split_name="Train")
    if val_img_src is not None and val_label_src is not None:
        if val_img_dist is None:
            val_img_dist = train_img_dist.replace("train", "val")
        if val_label_dist is None:
            val_label_dist = train_label_dist.replace("train", "val")
        create_dataset_subset(class_code_list, val_img_src, val_label_src, val_img_dist, val_label_dist,
                              class_code_to_info, split_name="Validation")
    else:
        for subdir in os.listdir(train_img_dist):
            src_img = os.path.join(train_img_dist, subdir)
            src_label = os.path.join(train_label_dist, subdir)
            dist_image = os.path.join(train_img_dist.replace("train", "val"), subdir)
            dist_label = os.path.join(train_label_dist.replace("train", "val"), subdir)
            train_test_split(src_img, src_label, dist_image, dist_label, test_size=val_split)
