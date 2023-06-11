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
    def __init__(self, prev_mask, num_objects, processor=None):
        super().__init__(None, prev_mask, None, None)
        self.K = num_objects
        self.processor = processor
        self.initial_mask = deepcopy(prev_mask).astype(np.uint8)
        self.drawn_mask = deepcopy(prev_mask).astype(np.uint8)
        self.current_mask = None
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths = [self.curr_path]

        self.size = None
        self.history.append(self.initial_mask)  # add the S0 into stack

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
        if mode == "draw":
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
   
        self.current_mask = self.drawn_mask # get the top of history and edit
        self.history.append(deepcopy(self.current_mask))

    def undo(self):
        _ = self.history.pop()
        self.current_mask = self.history[-1]
        self.all_paths = self.all_paths[:-2]
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)

        return self.current_mask

    def can_undo(self):
        return (len(self.history) > 2)
    
    def update(self):
        return self.current_mask

    def predict(self, data):
        predict_mask = self.processor.inference(data)
        
        # to store the predicted mask in same format as drawn_mask
        # use this to store the predicted mask in the history
        final_mask = np.zeros_like(self.drawn_mask)
        final_mask[0] = predict_mask
        if self.current_mask is None:
            self.current_mask = final_mask.copy()
        else:
            self.history.append(self.current_mask.copy())
            self.current_mask = self.drawn_mask.copy()
        return predict_mask

class BoxInteraction(Interaction):
    def __init__(self, prev_mask, num_objects, processor=None):
        super().__init__(None, prev_mask, None, None)
        self.K = num_objects
        self.processor = processor
        self.initial_mask = deepcopy(prev_mask).astype(np.uint8)
        self.current_mask = None
        
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths = [self.curr_path]

        self.init_x, self.init_y = 0, 0
        self.mode = None
        self.size = None
        self.history.append(self.initial_mask)  # add the S0 into stack

    def set_init_pos(self, x, y):
        self.init_x, self.init_y = x, y


    def set_size(self, size):
        self.size = size

    """
    k - object id
    vis - a tuple (visualization map, pass through alpha). None if not needed.
    """
    def push_point(self, x, y, k, vis=None, mode=None):
        self.mode = mode
        if vis is not None:
            vis_map, vis_alpha = vis
        selected = self.curr_path[k]
        selected.append((x, y))
        if len(selected) >= 2:
            # Plot visualization
            if vis is not None:
                # Visualization for drawing
                vis_map = cv2.rectangle(vis_map, 
                    (int(self.init_x), int(self.init_y)),
                    (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                    color_map[k], thickness=self.size)

                vis_alpha = cv2.rectangle(vis_alpha, 
                    (int(self.init_x), int(self.init_y)),
                    (int(round(selected[-1][0])), int(round(selected[-1][1]))),
                    0.75, thickness=self.size)
                

        if vis is not None:
            return vis_map, vis_alpha
    def end_path(self):
        
        print("end")
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)

        self.current_mask = np.logical_or(self.history[-1], self.get_mask()) # get the top of history and edit
        self.history.append(self.current_mask)
        # print("end", self.current_mask.shape, self.current_mask.dtype)
        # (1, 256, 256)  bool

    def undo(self):
        _ = self.history.pop()
    
        self.current_mask = self.history[-1]
        self.all_paths = self.all_paths[:-2]
        self.curr_path = [[] for _ in range(self.K + 1)]
        self.all_paths.append(self.curr_path)

        return self.current_mask

    def can_undo(self):
        return (len(self.history) > 2)
    
    def update(self):
        return self.current_mask
    
    def get_mask(self):
        # all_path structure =>  [path1, path2, []] , path2=> [[object0], [object1]], object1=>[(point pair init), ..., (point pair last)] 
        # so the unpack order is all_path[-2] (last path)[1] (object1)[0](first point pair)[0](x)
        # so the unpack order is all_path[-2] (last path)[1] (object1)[0](first point pair)[1](y)
        input_box = np.array([int(self.all_paths[-2][1][0][0]), int(self.all_paths[-2][1][0][1]), int(self.all_paths[-2][1][-1][0]), int(self.all_paths[-2][1][-1][1])])

        masks, _, _ = self.processor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,)

        return masks

        
