import argparse
from ultralytics import YOLO


def train_model(model_path, data, epochs, save_period, batch, **train_kwargs):
    model = YOLO(model_path)
    model.train(data=data, epochs=epochs, save_period=save_period, batch=batch, **train_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model configuration file")
    parser.add_argument("--data", type=str, required=True, help="Path to the data configuration file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--save_period", type=int, default=0, help="Save period for checkpoints")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (default: -1)")

    # Add a special argument to capture any additional keyword arguments
    parser.add_argument("--train_kwargs", nargs='*',
                        help="Additional keyword arguments in the format key1=value1 key2=value2 ...")

    args = parser.parse_args()

    # Parse the additional keyword arguments
    train_kwargs = dict(arg.split('=') for arg in args.train_kwargs) if args.train_kwargs else {}

    # Call the train_model function and pass all the arguments, including additional keyword arguments
    train_model(args.model_path, args.data, args.epochs, args.save_period, args.batch, **train_kwargs)
