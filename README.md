# YOLO-OpenImages-Segmentation

This repository provides a set of tools and scripts to download images from Google Open Images, format the dataset into YOLO format, create a YOLO configuration file, and train a YOLO model for object detection. Additionally, you can use the trained YOLO model for inference on new images.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)


## Folder Structure

The repository is organized as follows:

```/
├── config.yaml
├── create_yolo_config.py
├── dataset_formatter.py
├── dataset_utils/
│   ├── convert_to_yolo.py
│   ├── create_classes.py
│   ├── dataset_processing.py
│   ├── format_files.py
│   ├── split_data.py
├── download_dataset.py
├── predict.py
├── requirements.txt
├── train.py
```
## Installation

To get started with this project, follow these steps:

1. Clone this repository to your local machine:

```bash 
https://github.com/FaresElmenshawi/YOLO-OpenImages-Segmentation.git
```

2. Change your current directory to the cloned repository:

```bash
cd YOLO-OpenImages-Segmentation
```

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

1. Downloading a dataset
To download images from Google Open Images, use the `download_dataset.py` script with the following arguments:

```bash
python download_dataset.py --dataset_dir [DATASET_DIR] --max_samples [MAX_SAMPLES] --splits [SPLITS] --classes [CLASSES]
```
* --dataset_dir:  (default: parent directory of the current working directory): Directory where the dataset will be downloaded.
* --max_samples: Maximum number of samples to download.
* --splits: (default: ["train"]): Splits to download (e.g., train, validation).
* --classes: Classes to download.

2. Formatting the dataset

Use the dataset_formatter.py script with the following arguments to format the dataset:

```bash
python dataset_formatter.py --classes [CLASSES] --classes-csv [CLASSES_CSV] --working-dir [WORKING_DIR] --train-img-src [TRAIN_IMG_SRC] --train-label-src [TRAIN_LABEL_SRC] --train-img-dist [TRAIN_IMG_DIST] --train-label-dist [TRAIN_LABEL_DIST] --val-img-src [VAL_IMG_SRC] --val-label-src [VAL_LABEL_SRC] --val-img-dist [VAL_IMG_DIST] --val-label-dist [VAL_LABEL_DIST] --val-split [VAL_SPLIT] --ignore_warnings
```

* --classes (required): List of classes.
* --classes-csv (default: 'train/metadata/classes.csv'): Path to classes CSV file.
* --working-dir (default: parent directory of the current working directory): Working directory.
* --train-img-src (default: 'train/data'): Path to training image source directory.
* --train-label-src (default: 'train/labels/masks'): Path to training label source directory.
* --train-img-dist (default: 'images/train'): Path to training image destination directory.
* --train-label-dist (default: 'labels/train'): Path to training label destination directory.
* --val-img-src: Path to validation image source directory.
* --val-label-src: Path to validation label source directory.
* --val-img-dist: Path to validation image destination directory.
* --val-label-dist: Path to validation label destination directory.
* --val-split (default: 0.1): Validation split ratio.
* --ignore_warnings: Enable this flag to ignore warnings.

3. Creating YOLO Configuration
To create a YOLO configuration file, use the create_yolo_config.py script with the following arguments:

```bash
python create_yolo_config.py --data_dir [DATA_DIR] --train_path [TRAIN_PATH] --val_path [VAL_PATH] --names [NAMES]
```
* --data_dir (required): Path to data directory.
* --train_path (required): Path to train file.
* --val_path (required): Path to validation file.
* --names (required): Comma-separated list of names.

4. Training YOLO Model
To train your YOLO model with the formatted dataset and configuration file, use the train.py script with the following arguments:

```bash
python train.py --model_path [MODEL_PATH] --data [DATA] --epochs [EPOCHS] --save_period [SAVE_PERIOD] --batch [BATCH] --train_kwargs [TRAIN_KWARGS]
```

* --model_path (required): Path to the YOLO model configuration file.
* --data (required): Path to the data configuration file.
* --epochs (default: 1): Number of training epochs.
* --save_period (default: 0): Save period for checkpoints.
* --batch (default: -1): Batch size (default: -1).
* You can also provide additional keyword arguments using --train_kwargs in the format key1=value1 key2=value2 ...

## Credits

This project utilizes the following libraries:

[FiftyOne](https://docs.voxel51.com/) - A powerful library for dataset exploration and visualization.

[Ultralytics](https://github.com/ultralytics/ultralytics) - The YOLOv8 implementation used for training.

Special thanks to the contributors of these libraries for making this project possible.
