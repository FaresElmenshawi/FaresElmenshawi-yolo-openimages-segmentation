import os
from shutil import move

def train_test_split(src_img, src_label, dist_image, dist_label, test_size=0.1):
    """
    Split a dataset into training and testing subsets by moving image and label files.

    Args:
        src_img (str): Directory containing source images.
        src_label (str): Directory containing source labels.
        dist_image (str): Directory to move the testing images to.
        dist_label (str): Directory to move the testing labels to.
        test_size (float): The proportion of data to be used for testing (default is 0.1).
    """
    counter = 0
    limit = len(os.listdir(src_label)) * test_size
    for file_name in os.listdir(src_img):
        if counter < limit:
            if file_name.endswith(".jpg"):
                images_full_path = os.path.join(src_img, file_name)
                label_file_name = file_name.replace(".jpg", ".txt")
                labels_full_path = os.path.join(src_label, label_file_name)
                dest_full_image_path = os.path.join(dist_image, file_name)
                dest_full_label_path = os.path.join(dist_label, label_file_name)
                move(images_full_path, dest_full_image_path)
                move(labels_full_path, dest_full_label_path)
                counter += 1
            else:
                # Handle files that don't have a ".jpg" extension, if necessary
                pass
        else:
            break

if __name__ == "__main__":
    pass
    # train_test_split(src_img="C:/Users/Fares/fiftyone/open-images-v7/images/train/Football", src_label="C:/Users/Fares/fiftyone/open-images-v7/labels/train/Football"
    #                  , dist_image="C:/Users/Fares/fiftyone/open-images-v7/images/val/Football", dist_label="C:/Users/Fares/fiftyone/open-images-v7/labels/val/Football", test_size=0.1)
