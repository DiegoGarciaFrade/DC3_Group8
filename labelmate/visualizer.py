import os
import cv2
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from labelmate import CLASS_COLOR_MAPPING

logger = logging.getLogger(__name__)

def colorize_mask_labels(mask, color_mapping):
    """Convert masks with encoded labels to RGB scheme for visualization
    """
    mask_labels = mask[:,:,0][:]
    h, w = mask_labels.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for label, rgb in color_mapping.items():
        mask_rgb[mask_labels==label, :] = rgb

    return mask_rgb[:,:,:]

def visualize_output(
    experiment_name, 
    sample_id, 
    image_path, 
    mask_path, 
    prediction_path, 
    point_labels, 
    class_color_mapping=CLASS_COLOR_MAPPING, 
    ):
    """Display input image, ground truth mask and predicted mask
    """
    # read the original image
    image = cv2.imread(str(image_path.resolve()))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # read ground truth mask from specified path
    if os.path.exists(mask_path):
        gt_mask = cv2.imread(str(mask_path.resolve()))
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
        # colorize ground truth mask if it is label encoded
        if len(set(gt_mask.reshape(-1)) | set(class_color_mapping.keys())) == len(class_color_mapping):
            gt_mask = colorize_mask_labels(gt_mask, class_color_mapping)
    else:
        gt_mask = None

    # read the predicted mask from the specified path
    if os.path.exists(prediction_path):
        pred_mask = cv2.imread(str(prediction_path.resolve()))
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB)
        # colorize predicted mask if it is label encoded
        if len(set(pred_mask.reshape(-1)) | set(class_color_mapping.keys())) == len(class_color_mapping):
            pred_mask = colorize_mask_labels(pred_mask, class_color_mapping)
    else:
        pred_mask = None

    # get point labels for the given sample
    if point_labels is not None:
        if point_labels.shape[0] > 0:
            sample_point_labels_df = \
                point_labels\
                    .query(f"quadratid == '{sample_id}'")\
                    [['x', 'y', 'class_name']]
        else:
            sample_point_labels_df = pd.DataFrame({})
    else:
        sample_point_labels_df = pd.DataFrame({})

    # plot input image, ground truth mask and prediction side by side
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    ax[0].imshow(image)
    ax[0].set_title(f"{sample_id}")

    if sample_point_labels_df.shape[0] > 0:
        sns.scatterplot(data=sample_point_labels_df, x='x', y='y', hue='class_name', ax=ax[0], legend=False)

    if gt_mask is not None:
        ax[1].imshow(gt_mask)
        ax[1].set_title("GT Mask")

    if pred_mask is not None:
        ax[2].imshow(pred_mask)
        ax[2].set_title(f"{experiment_name}")

    plt.show()