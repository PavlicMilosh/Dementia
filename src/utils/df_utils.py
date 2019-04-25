from pandas import DataFrame


def split_by_classes(df: DataFrame, y_col):
    retval = {}
    classes = df[y_col].unique()

    for clazz in classes:
        retval[clazz] = df.loc[df[y_col] == clazz]

    return retval