from src.data_preprocessing.labels import Label


def separate_datasets(df, label_col):
    diagnosed_df = df.copy()
    healthy_df = df.copy()

    diagnosed = [
        Label.UNCERTAIN.fullname,
        Label.AD.fullname,
        Label.NON_AD.fullname
    ]

    d = {i: Label.DIAGNOSED.fullname for i in diagnosed}

    diagnosed_df = diagnosed_df.loc[diagnosed_df[label_col] != Label.COGNITIVELY_NORMAL.fullname]
    healthy_df[label_col].replace(d, inplace=True)

    return healthy_df, diagnosed_df
