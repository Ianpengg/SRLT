import numpy as np
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QKeySequence, QImage, QTextCursor
from PyQt5.QtWidgets import (QWidget, QApplication, QComboBox, 
    QHBoxLayout, QLabel, QPushButton, QTextEdit, 
    QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, 
    QShortcut, QRadioButton, QProgressBar, QFileDialog)
from PyQt5.QtCore import Qt, QTimer 

from .ui import Ui_main_widget
from .interact.timer import Timer
from .interact.interaction import * 
from .utils.gui_utils import overlay_moving_mask
from .utils.file_utils import *

class MainWindow_controller(QtWidgets.QWidget):
    def __init__(self, images):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        
        self.cursor = 0 # represent the current frame index
        self.waiting_to_start = True
        
        self.images = images
        self.num_frames = len(self.images)
        self.radar_data = np.load(self.images[self.cursor], allow_pickle=True).item()['raw_radar_0']
        self.width, self.height = self.radar_data.shape


        self.ui = Ui_main_widget()
        self.ui.setupUi(self)
        self.setLayout(self.ui.layout)
        self.setWindowTitle("SR Label")
        self.setGeometry(2000, 100, self.width, self.height+100)
        
        self.setup_control()

        self.ui.tl_slider.setMaximum(self.num_frames-1)
        #self.resize(self.width, self.height) 
        # initialize
        self.viz_mode = 'davis'
        
        self.current_mask = np.zeros((self.num_frames, self.height, self.width), dtype=np.uint8)
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.vis_hist = deque(maxlen=100)
        self.cursur = 0
        self.on_showing = None


        # the class of the drawing functions ex 
        self.interaction = None 
        self.global_timer = Timer().start()
        self.algo_timer = Timer()
        self.user_timer = Timer()
        self.console_push_text('Initialized.')
        self.zoom_pixels = 100
        self.pressed = False
        self.right_click = False
        self.ctrl_size = False
        self.current_object = 1
        self.last_ex = self.last_ey = 0
        #print(self.ui.main_canvas.size())
        self.showCurrentFrame()

    def setup_control(self, ):
        self.ui.tl_slider.valueChanged.connect(self.tl_slide)
        self.ui.brush_size_bar.valueChanged.connect(self.brush_slide)
        self.ui.main_canvas.mousePressEvent = self.on_press 
        self.ui.timer.timeout.connect(self.on_time)


    def resizeEvent(self, event):
        self.showCurrentFrame()
        
    def on_time(self):
        self.cursor += 1
        if self.cursor > self.num_frames-1:
            self.cursor = 0
        self.ui.tl_slider.setValue(self.cursor)

    def console_push_text(self, text):
        text = '[A: %s, U: %s]: %s' % (self.algo_timer.format(), self.user_timer.format(), text)
        self.ui.console.appendPlainText(text)
        self.ui.console.moveCursor(QTextCursor.End)
        print(text)


    def brush_slide(self):
        self.brush_size = self.ui.brush_size_bar.value()
        self.ui.brush_size_label.setText('Brush size: %d' % self.brush_size)
        self.interaction.set_size(self.brush_size)
        

    def tl_slide(self):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            self.console_push_text('Timers started.')

        #self.reset_this_interaction()
        self.cursor = self.ui.tl_slider.value()
        self.showCurrentFrame()
    
    
    def compose_current_im(self):
         self.viz = overlay_moving_mask(self.images[self.cursur])
         #print(self.viz.shape) 


    def update_minimap(self):
        # Limit it within the valid range
        ex, ey = self.last_ex, self.last_ey
        r = self.zoom_pixels//2
        ex = int(round(max(r, min(self.width-r, ex))))
        ey = int(round(max(r, min(self.height-r, ey))))

        patch = self.viz_with_stroke[ey-r:ey+r, ex-r:ex+r, :].astype(np.uint8)

        height, width, channel = patch.shape
        bytesPerLine = 3 * width
        qImg = QImage(patch.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.ui.minimap.setPixmap(QPixmap(qImg.scaled(self.ui.minimap.size(),
                Qt.KeepAspectRatio, Qt.FastTransformation)))


    def update_interact_vis(self):
        # Update the interactions without re-computing the overlay
        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        vis_map = self.vis_map
        vis_alpha = self.vis_alpha
        brush_vis_map = self.brush_vis_map
        brush_vis_alpha = self.brush_vis_alpha

        self.viz_with_stroke = self.viz*(1-vis_alpha) + vis_map*vis_alpha
        self.viz_with_stroke = self.viz_with_stroke*(1-brush_vis_alpha) + brush_vis_map*brush_vis_alpha
        self.viz_with_stroke = self.viz_with_stroke.astype(np.uint8)

        qImg = QImage(self.viz_with_stroke.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.ui.main_canvas.setPixmap(QPixmap(qImg.scaled(self.ui.main_canvas.size(),
                Qt.KeepAspectRatio, Qt.FastTransformation)))
        
        self.main_canvas_size = self.ui.main_canvas.size()
        self.image_size = qImg.size()


    def showCurrentFrame(self):
        self.compose_current_im()
        self.update_interact_vis()
        self.update_minimap()
        self.ui.frame_log.setText(f"{self.cursor}/ {self.num_frames-1}")
        self.ui.tl_slider.setValue(self.cursor)

    def setBrushSize(self):
        self.ui.brush_size_label.setText(f"Brush Size: {self.ui.brush_size_bar.value()}")
    
    
    def get_scaled_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh/oh
        w_ratio = nw/ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh/dominate_ratio, nw/dominate_ratio
        x -= (fw-ow)/2
        y -= (fh-oh)/2

        x = max(0, min(self.width-1, x))
        y = max(0, min(self.height-1, y))

        # return int(round(x)), int(round(y))
        return x, y


    def on_press(self, event):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            self.console_push_text('Timers started.')
    
        self.user_timer.pause()
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.pressed = True
        self.right_click = (event.button() != 1)
        self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))
        # Ordinary interaction (might be in local mode)
        
        if self.interaction is None:
            if len(self.this_frame_interactions) > 0:
                prev_soft_mask = self.this_frame_interactions[-1].out_prob
            else:
                prev_soft_mask = self.processor.prob[1:, self.cursur]
        else:
            # Not used if the previous interaction is still valid
            # Don't worry about stacking effects here
            prev_soft_mask = self.interaction.out_prob
        prev_hard_mask = self.processor.masks[self.cursur]
        image = self.processor.images[:,self.cursur]
        h, w = self.height, self.width

        last_interaction = self.interaction
        new_interaction = None
        if self.curr_interaction == 'Free':
            if last_interaction is None or type(last_interaction) != FreeInteraction:
                self.complete_interaction()
                new_interaction = FreeInteraction(image, prev_soft_mask, (h, w), 
                            self.num_objects, self.processor.pad)
                new_interaction.set_size(self.brush_size)

        if new_interaction is not None:
            self.interaction = new_interaction

        # Just motion it as the first step
        #self.on_motion(event)
        self.user_timer.start()

if __name__ == "__main__":
    pass
