from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, SVMSMOTE


def get_oversampler(sampler_name, **add_params):
    sampler_name = sampler_name.lower()

    if sampler_name == 'adasyn':
        return ADASYN(**add_params)
    elif sampler_name == 'smote':
        return SMOTE(**add_params)
    elif sampler_name == 'smotenc':
        return SMOTENC(**add_params)
    elif sampler_name == 'svmsmote':
        return SVMSMOTE(**add_params)
    else:
        print('Choose one of predefined over-samplers')
