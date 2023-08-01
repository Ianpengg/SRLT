import numpy as np


def overlay_moving_mask(image, mask, alpha=0.6):
    """
    Overlay segmentation on top of RGB image.
    """
    # Create the mask for 3 channel RGB image
    mask_1 = np.zeros((mask.shape[0], mask.shape[1]))
    mask_1[mask == 1] = 1
    mask_2 = np.zeros((mask.shape[0], mask.shape[1]))
    mask_2[mask == 2] = 1
    print("mask", mask_1.shape, mask_2.shape)
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    print("overlap", np.unique(mask))
    colored_mask[:, :, 0] = (mask_1 * 255).astype(np.uint8)
    colored_mask[:, :, 1] = (mask_2 * 255).astype(np.uint8)

    # overlap the image and the mask
    overlap_image = image * alpha + (1 - alpha) * colored_mask

    return overlap_image.astype(image.dtype)
