from pandas import DataFrame

from src.data_preprocessing.basic_preprocessing import load_data


def split_by_classes(df: DataFrame, y_col):
    retval = {}
    classes = df[y_col].unique()

    for clazz in classes:
        retval[clazz] = df.loc[df[y_col] == clazz]

    return retval