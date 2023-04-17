"""
Contains all the types of interaction related to the GUI
Not related to automatic evaluation in the DAVIS dataset

You can inherit the Interaction class to create new interaction types
undo is (sometimes partially) supported
"""

#import torch
import numpy as np
import cv2
import time
from collections import deque
from copy import deepcopy

color_map = [
    [0, 0, 0], 
    [255, 50, 50], 
    [50, 255, 50], 
    [50, 50, 255], 
    [255, 50, 255], 
    [50, 255, 255], 
    [255, 255, 50], 
]

max_history = 50

class Interaction:
    def __init__(self, image, prev_mask, true_size, controller):
        self.image = image
        self.prev_mask = prev_mask

        self.start_time = time.time()
        self.history = deque(maxlen=max_history)

    def undo(self):
        pass

    def can_undo(self):
        return len(self.history) > 0

    def update(self):
        pass

class FreeInteraction(Interaction):
    def __init__(self, prev_mask, initial_mask, num_objects, processor=None):
        super().__init__(None, prev_mask, None, None)
        self.K = num_objects
        self.processor = processor
        self.drawn_mask = deepcopy(prev_mask).astype(np.uint8)
        # self.drawn_mask[0] = initial_mask
        self.current_mask = None
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths = [self.curr_path]

        self.size = None
        self.surplus_history = False

    def set_size(self, size):
        self.size = size

    """
    k - object id
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """
    def push_point(self, x, y, k, vis=None, mode=None):
        if vis is not None:
            vis_map, vis_alpha = vis
        selected = self.curr_path[k]
        selected.append((x, y))
        if len(selected) >= 2:
            for i in range(self.K):
                if mode == "draw":
                    self.drawn_mask[i] = cv2.line(self.drawn_mask[i], (int(round(selected[-2][0])), int(round(selected[-2][1]))), \
                    (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                    int((i+1)==k), thickness=self.size)
                elif mode == "erase":
                    self.drawn_mask[i] = cv2.line(self.drawn_mask[i], (int(round(selected[-2][0])), int(round(selected[-2][1]))), \
                    (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                    color_map[0], thickness=self.size)
            # Plot visualization
            if vis is not None:
                # Visualization for drawing
                if k == 0:
                    vis_map = cv2.line(vis_map, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        color_map[k], thickness=self.size)
                else:
                    vis_map = cv2.line(vis_map, 
                        (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                        (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                        color_map[k], thickness=self.size)
                # Visualization on/off boolean filter
                vis_alpha = cv2.line(vis_alpha, 
                    (int(round(selected[-2][0])), int(round(selected[-2][1]))),
                    (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                    0.75, thickness=self.size)

        if vis is not None:
            return vis_map, vis_alpha

    def end_path(self):
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)
        if self.current_mask is None:
            self.current_mask = self.drawn_mask.copy()
        else:
            self.history.append(self.current_mask.copy())
            self.current_mask = self.drawn_mask.copy()

    def undo(self):

        self.current_mask = self.history.pop()
        self.all_paths = self.all_paths[:-2]
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)

        return self.current_mask

    def can_undo(self):
        return (len(self.history) > 0)
    
    def update(self):
        return self.drawn_mask

    def predict(self, data):
        predict_mask = self.processor.inference(data)
        
        # to store the predicted mask in same format as drawn_mask
        # use this to store the predicted mask in the history
        final_mask = np.zeros_like(self.drawn_mask)
        final_mask[0] = predict_mask
        if self.current_mask is None:
            self.current_mask = self.final_mask.copy()
        else:
            self.history.append(self.current_mask.copy())
            self.current_mask = self.drawn_mask.copy()
        return predict_mask