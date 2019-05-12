import os.path as osp


# CREATING IMAGE DATASET FROM MRI ======================================================================================

NEUROIMAGE_DPI = 128
X_CUTS = 30
Y_CUTS = 30
Z_CUTS = 30
SMOOTHING_LEVEL = 0
FORMAT = 'png'
BLACK_BACKGROUND = True
MODE = 'anat' # anat (anatomic) or epi (echo planar imaging)

# MODEL TRAINING =======================================================================================================

CV_FOLDS = 10


# RANDOM STATE =========================================================================================================

RANDOM_STATE = 420


# ROOTS DIRS ===========================================================================================================

BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))


# DATA DIR =============================================================================================================

NEUROIMAGING_DIR = osp.join(BASE_DIR, 'data', 'neuroimaging_data')
STRUCTURED_DIR = osp.join(BASE_DIR, 'data', 'numerical_data')


# MODEL ================================================================================================================

MODELS_DIR = osp.join(BASE_DIR, 'model')
