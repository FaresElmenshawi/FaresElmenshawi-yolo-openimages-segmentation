import argparse
import yaml


def create_yolo_config(data_dir, train_path, val_path, names):
    data = {
        "path": data_dir,
        "train": train_path,
        "val": val_path,
        "names": names
    }

    with open("config.yaml", "w") as yaml_file:
        yaml.dump(data, yaml_file)


def main():
    parser = argparse.ArgumentParser(description="Create YOLO configuration file")

    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--train_path", required=True, help="Path to train file")
    parser.add_argument("--val_path", required=True, help="Path to validation file")
    parser.add_argument("--names", required=True, help="Comma-separated list of names")

    args = parser.parse_args()

    names_list = args.names.split(',')
    create_yolo_config(args.data_dir, args.train_path, args.val_path, names_list)


if __name__ == "__main__":
    main()
