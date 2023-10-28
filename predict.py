import cv2
import math
import numpy as np
from ultralytics import YOLO
from seaborn import color_palette

model = YOLO("best.pt")

img = cv2.imread('C:/Users/Fares/Desktop/Segmentation/images/train/5f8b855be9d2b964.jpg')
mask = np.zeros_like(img)
# mask = mask.reshape((-1, 1, 2)).astype(np.int32)
results = model(img)
print(results[0].masks.xy[0])
# for points in results[0].masks.xy:
#     print(points.shape)
#     cv2.fillPoly(mask, [points.reshape((-1, 1, 2)).astype(np.int32)], (0, 0, 255))  # Fill with white color
# result = cv2.addWeighted(img, 1, mask, 0.5, 0)


# cv2.imshow("img", result)
# cv2.waitKey(0)


# mask = np.zeros_like(img)
#
# points = results[0].masks.xy[0]
# points = points.reshape((-1, 1, 2)).astype(np.int32)
# cv2.fillPoly(mask, [points], (0, 0, 255))  # Fill with white color
# result = cv2.addWeighted(img, 1, mask, 0.5, 0)
# cv2.imshow("img", result)
# cv2.waitKey(0)
# cv2.destro

#
# def draw_segmentation(frame, masks, classes, colors):
#     for points in masks:
#         points = points.reshape((-1, 1, 2)).astype(np.int32)
#         cv2.fillPoly(mask, [points], (0, 0, 255))  # Fill with white color
#         cls = int(box.cls[0])
#         class_name = class_names[cls]
#
#         # Retrieving the color for the class
#         color = colors[cls]
#
#
# def draw_bbox(frame,masks, boxes, class_names, colors):
#     """
#     Draws bounding boxes with labels on the input frame.
#
#     Args:
#         frame (numpy.ndarray): The input image frame.
#         boxes (List[Object]): List of bounding boxes.
#         class_names (List[str]): List of class names.
#         colors (List[Tuple[int]]): List of RGB color values.
#
#     Returns:
#         None
#     """
#     # for points in masks:
#     #     points = points.reshape((-1, 1, 2)).astype(np.int32)
#     #     cv2.fillPoly(mask, [points], (0, 0, 255))  # Fill with white color
#     #     cls = int(box.cls[0])
#     #     class_name = class_names[cls]
#
#     for box in boxes:
#         x1, y1, x2, y2 = box.xyxy[0]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#         # Extracting the class label and name
#         cls = int(box.cls[0])
#         class_name = class_names[cls]
#
#         # Retrieving the color for the class
#         color = colors[cls]
#
#         # Drawing the bounding box on the image
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#
#         # Formatting the confidence level and label text
#         conf = math.ceil((box.conf[0] * 100)) / 100
#         label = f'{class_name} ({conf}%)'
#
#         # Calculating the size of the label text
#         text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#         # Calculating the coordinates for the background rectangle of the label
#         rect_coords = x1 + text_size[0], y1 - text_size[1] - 3
#
#         # Drawing the background rectangle and the label text
#         cv2.rectangle(frame, (x1, y1), rect_coords, color, -1, cv2.LINE_AA)
#         cv2.putText(frame, label, (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
#
#
#
#
# def run_yolo(model, source=0, prediction_type='video'):
#     """
#     Performs object detection on an image or video.
#
#     Args:
#         model_name (str): The name of the model to use for object detection. Default is 'yolov8s.pt'.
#         source (Union[str, int]): The path to the image or video file or webcam index. Default is 0 (webcam).
#         prediction_type (str): The type of prediction to make. Valid values are 'image' and 'video'. Default is 'video'.
#         class_path (str): The path to the file containing class names. Default is 'classes.txt'.
#         outdir (str): The output directory or file name. Default is 'output'.
#
#     Returns:
#         None
#     """
#     # Loading the class names from the file
#     class_names = ['Person']
#     n_classes = len(class_names)
#     #  Generating colors for each class
#     colors = {}
#     #  Generate a color palette
#     for i in range(n_classes):
#         color = tuple((np.array(color_palette('hls', n_classes)) * 255)[i])
#         colors[i] = color
#
#     # Checking the prediction type
#     if prediction_type == 'video':
#         # Capturing the video from the source
#         cap = cv2.VideoCapture(source)
#
#         while True:
#             # Reading a frame from the video
#             ret, frame = cap.read()
#
#             # Performing object detection on the frame
#             results = model(frame, stream=True, conf=0.5, verbose=False)
#
#             # Iterating over the detected objects
#             for i, result in enumerate(results):
#                 # Extracting the bounding box coordinates
#                 boxes = result.boxes
#                 draw_bbox(frame, boxes, class_names, colors)
#
#                 cv2.imshow("Image", frame)
#
#                 # Checking if the user pressed 'q' to exit the loop
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#
#     elif prediction_type == 'image':
#         # Appending '.jpg' extension to the output directory
#         frame = cv2.imread(source)
#         results = model(frame, stream=True, conf=0.5)
#
#         # Iterating over the detected objects
#         for i, result in enumerate(results):
#             # Extracting the bounding box coordinates
#             boxes = result.boxes
#
#             draw_bbox(frame, boxes, class_names, colors)
#             cv2.imshow("Image", frame)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#


# if __name__ == '__main__':
#     model = YOLO("yolov8n-seg.pt")
#     run_yolo(model, source='img.jpg', prediction_type='image')
