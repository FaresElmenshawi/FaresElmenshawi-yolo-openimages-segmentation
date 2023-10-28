import pandas as pd


class LabelNotFoundError(Exception):
    """
    Exception raised when a label is not found in the classes DataFrame.

    """
    pass


def find_label(classes_df, label):
    """
    Find the code corresponding to a label in the classes DataFrame.

    Args:
        classes_df (pd.DataFrame): A DataFrame containing class codes and labels.
        label (str): The label to search for.

    Returns:
        str: The code corresponding to the label.

    Raises:
        LabelNotFoundError: If the label is not found in the DataFrame.
    """
    result = classes_df[classes_df[1] == label]
    if not result.empty:
        code = result.iloc[0, 0]
        return code.replace("/", "")
    else:
        raise LabelNotFoundError(label)


def extract_classes(label_names, classes_csv_path="train/metadata/classes.csv"):
    """
    Extract class information from a CSV file and create a mapping of class codes to class names.

    Args:
        classes_csv_path (str): The path to the CSV file containing class information.
        label_names (list): A list of label names to extract.

    Returns:
        list: A list of class codes.
        dict: A dictionary mapping class codes to class information.

    """
    classes_df = pd.read_csv(classes_csv_path, header=None)
    code_list = []
    code_to_info = {}
    for i, label in enumerate(label_names):
        code = find_label(classes_df, label)
        code_to_info[code] = [i, label]
        code_list.append(code)
    return code_list, code_to_info
