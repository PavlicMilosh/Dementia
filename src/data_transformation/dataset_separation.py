def separate_datasets(X, y):
    df = X.copy()
    df['dx1'] = y
    diagnosed = ['Uncertain Dementia', 'Alzheimer Dementia', 'Non AD Dementia']
    d = {i: 'Diagnosed' for i in diagnosed}
    diagnosed_df = df.loc[df['dx1'] != 'Cognitively Normal']
    healthy_df = df.map(d)
    return healthy_df, diagnosed_df
