import numpy as np


def overlay_moving_mask(image, mask, alpha=0.6):
    """
    Overlay segmentation on top of RGB image.
    """
    # Create the mask for 3 channel RGB image
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    colored_mask[:, :, 0] = (mask * 255).astype(np.uint8)

    # overlap the image and the mask
    overlap_image = image * alpha + (1 - alpha) * colored_mask

    return overlap_image.astype(image.dtype)
