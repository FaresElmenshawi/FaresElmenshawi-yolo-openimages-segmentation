import cv2


def convert_to_yolo(mask_path, output_dir, label_idx):
    """
    Convert a binary mask image to YOLO format annotation and save it to a text file.

    Args:
        mask_path (str): The path to the binary mask image file.
        output_dir (str): The path to the output YOLO annotation text file.
        label_idx (int): The label index to assign to the annotation.

    This function reads a binary mask image, extracts its contours, converts the contours to polygons, and
    saves the YOLO format annotation to the specified text file.

    YOLO format annotation consists of lines with the following format:
    "<label_idx> <center_x> <center_y> <width> <height>"
    where center_x and center_y are the coordinates of the object's center, and width and height are the object's dimensions.
    """
    # Load the binary mask and get its contours
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    h, w = mask.shape
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the contours to polygons
    polygons = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 300:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / w)
                polygon.append(y / h)
            polygons.append(polygon)

    # Save the polygons to the output YOLO annotation file
    with open(output_dir, 'w') as f:
        for polygon in polygons:
            for idx, point in enumerate(polygon):
                if idx == len(polygon) - 1:
                    f.write('{}\n'.format(point))
                elif idx == 0:
                    f.write('{} {} '.format(label_idx, point))
                else:
                    f.write('{} '.format(point))
