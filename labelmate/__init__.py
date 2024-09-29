import logging
import numpy as np

# define a formatter to display the messages to console (standard output)
console_formatter = logging.Formatter('%(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)

# define a logger for this package and attach the console handler
logger = logging.getLogger('labelmate')
# logger.handlers.clear()
logger.propagate = False
logger.addHandler(console_handler)

# set an appropriate level of logging for this package
logger.setLevel(logging.DEBUG)

# Global Variables

# Black (0,0,0): Others, Red (255,0,0): Hard Coral, Blue (0,0,255): Soft Coral
CLASS_COLOR_MAPPING = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 0, 255]}
CLASS_NAME_MAPPING = {0: 'Others', 1: 'Hard Coral', 2: 'Soft Coral'}

# Reef Support dense masks have interior and contour colors that are different
REEF_SUPPORT_COLORS = {
    'K': [0, 0, 0],
    'R': [255, 0, 0], 'Y': [255, 255, 0],
    'B': [0, 0, 255], 'O': [255, 165, 0],
}
REEF_SUPPORT_COLOR_MAPPING = {
    'K': 0,
    'R': 1, 'Y': 1,
    'B': 2, 'O': 2,
}

def encode_mask_with_contours(mask, colors=REEF_SUPPORT_COLORS, color_mapping=REEF_SUPPORT_COLOR_MAPPING):
    """Convert dense ground truth masks with contours to label encoding format
    """
    mask_encoded = np.zeros(mask.shape, dtype=int)
    for color_code, label in color_mapping.items():
        mask_encoded[np.all(mask == colors[color_code], axis=-1), :] = label
    return mask_encoded[:,:,:]