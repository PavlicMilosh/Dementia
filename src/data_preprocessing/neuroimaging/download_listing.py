import os.path as osp

import numpy as np
import pandas as pd

from settings.settings import NEUROIMAGING_DIR
from src.data_preprocessing.labels import LABEL_MAP
from src.data_preprocessing.util import adjust_target_variable_labels

# portions to download (0-1)
PORTION_MAP = {
    "Cognitively Normal": 0.3,
    "Uncertain Dementia": 1,
    "Alzheimer Dementia": 1,
    "Non AD Dementia":    1
}


def create_download_list(path, save_path):
    """
    Creates a csv with download list, since there is more than 3.5TB of
    neuroimaging data we need to filter images we need
    :param path: path to list of all MRI scans (csv file)
    :return:
    """

    df = pd.read_csv(path)
    adjust_target_variable_labels(df)

    portions = []
    for target_label, df_dx1 in df.groupby('dx1'):

        if target_label not in LABEL_MAP.keys():
            continue

        portion = PORTION_MAP[target_label]
        msk = np.random.rand(len(df_dx1)) < portion
        portions.append(df_dx1[msk])

    selected_df = pd.concat(portions)
    selected_df = selected_df['MR ID']

    with open(save_path, 'w'):
        for selected in selected_df['MR ID']:
            pass

    # selected_df.to_csv(save_path, index=False)


if __name__ == '__main__':
    create_download_list(osp.join(NEUROIMAGING_DIR, 'list_of_all_mri_scans.csv'),
                         osp.join(NEUROIMAGING_DIR, 'list_of_selected_mri_scans.csv'))

