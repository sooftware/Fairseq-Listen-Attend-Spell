import os
import importlib

DEFAULT_ENC_VGGBLOCK_CONFIG = ((64, 3, 2, 2, True), (128, 3, 2, 2, True))
DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        importlib.import_module('fairseq_las.models.%s' % model_name)
