from pandas import DataFrame

from src.data_preprocessing.labels import LABEL_MAP


def adjust_target_variable_labels(df: DataFrame, inplace=True):
    """
    Adjusts target variables based on LABEL MAP in src/data_preprocessing/labels.py
    :param df: target data frame
    :param inplace: If inplace is True, modify given data frame, else return new data frame
    """
    label_map = {}
    for key, value in LABEL_MAP.items():
        for v in value:
            label_map[v] = key

    # Replace values in DataFrame
    if inplace:
        df.replace({'dx1': label_map}, inplace=True)
    else:
        return df.replace({'dx1': label_map}, inplace=False)
