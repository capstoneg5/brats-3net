# config/__init__.py
from .paths import paths
from .model_config import model_config
from .training import training_config
from .brats import brats_config
BRATS_MODALITIES = brats_config.BRATS_MODALITIES