import cv2
import numpy as np
from ultralytics import YOLO
from seaborn import color_palette
import math
import os
import argparse

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']


def load_class_names(file_name):
    """
    Returns a list of class names read from the file `file_name`.

    Args:
        file_name (str): The path to the file containing the class names.

    Returns:
        List[str]: A list of class names.
    """
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def generate_color_palette(n_classes):
    """
    Generate a color palette for visualizing different classes.

    Args:
        n_classes (int): The number of classes.

    Returns:
        dict: A dictionary of class indices mapped to RGB colors.
    """
    colors = {i: tuple(np.array(color_palette('hls', n_classes)[i]) * 255) for i in range(n_classes)}
    return colors


def draw_and_mask_objects(frame, results, class_names, colors, draw_bboxes=True, draw_masks=True):
    """
    Draw bounding boxes and masks on an image.

    Args:
        frame (np.ndarray): The input image to draw on.
        results (ultralytics.engine.results.Results): Detected objects and their attributes.
        class_names (list of str): List of class names.
        colors (dict): A dictionary of class indices mapped to RGB colors.
        draw_bboxes (bool, optional): Whether to draw bounding boxes. Defaults to True.
        draw_masks (bool, optional): Whether to draw masks. Defaults to True.

    Returns:
        np.ndarray: The image with annotations.
    """
    if draw_masks:
        mask = np.zeros_like(frame, dtype=np.uint8)

    if results is not None and results.boxes is not None and results.masks is not None:
        for confidence, box, cls, mask_coords in zip(results.boxes.conf, results.boxes.xyxy, results.boxes.cls,
                                                     results.masks.xy):
            x1, y1, x2, y2 = map(int, box.tolist())
            cls = int(cls)
            class_name = class_names[cls]
            color = colors[cls]

            if draw_bboxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            conf = math.ceil((confidence.cpu() * 100)) / 100
            label = f'{class_name} ({conf}%)'

            text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            rect_coords = x1 + text_size[0], y1 - text_size[1] - 3

            cv2.rectangle(frame, (x1, y1), rect_coords, color, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            if draw_masks:
                mask_coords = mask_coords.astype(int)
                mask_coords = mask_coords.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [mask_coords], color)
                frame = cv2.addWeighted(frame, 0.9, mask, 0.5, 0)

    return frame


def decide_inference_type(src):
    """
    Determine the type of inference (image or video) based on the input source.

    Args:
        src (str or int): Input source, either a file path or camera index.

    Returns:
        str: 'image' for image source, 'video' for video source.
    """
    if isinstance(src, int):
        return 'video'
    file_extension = os.path.splitext(src)[-1].lower()
    if file_extension in VIDEO_EXTENSIONS:
        return "video"
    else:
        return "image"


def perform_inference(src, model_path='yolov8n-seg.pt', classes_path='coco.txt', verbose=False, conf=0.3,
                      draw_masks=True, draw_bboxes=True):
    """
    Perform object detection and segmentation on an image or video source.

    Args:
        src (str or int): Input source, either a file path or camera index.
        model_path (str, optional): Path to the YOLO model file. Defaults to 'yolov8n-seg.pt'.
        classes_path (str, optional): Path to the class names file. Defaults to 'coco.txt'.
        verbose (bool, optional): Whether to display additional information during inference. Defaults to False.
        conf (float, optional): Confidence threshold for object detection. Defaults to 0.3.
        draw_masks (bool, optional): Whether to draw masks. Defaults to True.
        draw_bboxes (bool, optional): Whether to draw bounding boxes. Defaults to True

    Returns:
        None
    """
    model = YOLO(model_path)
    classes = load_class_names(classes_path)
    n_classes = len(classes)
    colors = generate_color_palette(n_classes)
    inference_type = decide_inference_type(src)
    if inference_type == 'video':
        cap = cv2.VideoCapture(src)

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            if not ret:
                break

            results = model(frame, verbose=verbose, conf=conf)
            for result in results:
                frame = draw_and_mask_objects(frame, result, class_names=classes, colors=colors, draw_masks=draw_masks,
                                              draw_bboxes=draw_bboxes)

            cv2.imshow("Segmentation Results", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(src)
        results = model(img)
        img = draw_and_mask_objects(img, results[0], class_names=classes, colors=colors,
                                    draw_masks=draw_masks, draw_bboxes=draw_bboxes)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    # Create a command-line argument parser for YOLOv8 Inference
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")

    # Define input source argument
    parser.add_argument("--src", help="Specify the input source, which can be a file path or camera index")

    # Define YOLO model configuration file argument
    parser.add_argument("--model_path", type=str, default='yolov8n-seg.pt',
                        help="Path to the YOLO model configuration file")

    # Define class names file argument
    parser.add_argument("--classes_path", type=str, default='coco.txt',
                        help="Path to the file containing class names")

    # Define verbosity option
    parser.add_argument("--verbose", action="store_true",
                        help="Enable this option to display the model's output during inference")

    # Define confidence threshold for detection and segmentation
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Set the confidence threshold to initiate detection and segmentation")

    # Enable or disable drawing segmentation masks
    parser.add_argument("--no_masks", action="store_false", help="Don't draw Segmentation mask (default: False)")

    # Enable or disable drawing bounding boxes
    parser.add_argument("--no_bboxes", action="store_false", help="Don't draw bounding boxes (default: False)")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if 'src' is a digit (camera index), and if so, convert it to an integer
    if args.src and args.src.isdigit():
        args.src = int(args.src)

    # Perform inference using the provided arguments
    perform_inference(args.src, args.model_path, args.classes_path, args.verbose, args.conf, args.no_masks,
                      args.no_bboxes)
