from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler


def get_imputer(imputer_name, **add_params):

    imputer_name = imputer_name.lower()

    if imputer_name == 'knn':
        return KNN(**add_params)
    elif imputer_name.lower() == 'nnm':
        return NuclearNormMinimization(**add_params)
    elif imputer_name == 'soft':
        return SoftImpute(**add_params)
    elif imputer_name == 'iterative':
        return IterativeImputer(**add_params)
    elif imputer_name == 'biscaler':
        return BiScaler(**add_params)
    else:
        print('Choose one of predefined imputers')

