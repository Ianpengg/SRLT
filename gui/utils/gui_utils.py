import numpy as np




def overlay_moving_mask(image, alpha=0.5):
     """ Overlay segmentation on top of RGB image. from davis official"""
     data = np.load(image, allow_pickle=True).item()
     radar = data['raw_radar_0']
     mask = data['gt_moving'].astype(np.uint8)
     im_overlay = np.stack((radar, radar, radar), axis=2)
     
     #colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
     #colored_mask[:, :, 2] = mask
     color_map_np = [[0, 0, 255], [255, 255, 0]]
     colored_mask = np.array(color_map_np)[mask]
     foreground = im_overlay*alpha + (1-alpha)*colored_mask
     binary_mask = (mask > 0)
     # Compose image
     im_overlay[binary_mask] = foreground[binary_mask]
     #cv2.imshow("ets", im_overlay)
     #cv2.waitKey()
     return im_overlay.astype(im_overlay.dtype)
