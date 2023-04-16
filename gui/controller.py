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
from .btn_controller import ButtonController


class MainWindow_controller(QtWidgets.QWidget):
    def __init__(self, files_path, timestamps):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        
        # Set up some initial values
        self.mode = "test"
        self.cursor = 2 # represent the current frame index
        self.radar_timestamps = timestamps 
        self.files_path = files_path
        self.num_frames = len(self.radar_timestamps)
        self.radar_data = load_data(files_path, self.cursor)
        self.image = load_image(files_path, self.cursor) 
        self.mask = load_mask(files_path, self.cursor)
        self.width, self.height = self.image.shape[:2]

        self.ui = Ui_main_widget()
        self.ui.setupUi(self)
        self.setLayout(self.ui.layout)
        self.setWindowTitle("SR Label")
        self.setGeometry(2000, 100, self.width, self.height+100)

        # setup the slider tick and minimum accroding to the train/test split
        # for train the tick = 2, for test tick = 1
        if self.mode == "train":
            self.ui.tl_slider.setSingleStep(2)
        elif self.mode == "test":
            self.ui.tl_slider.setSingleStep(1)
        self.ui.tl_slider.setMinimum(self.cursor)
        self.ui.tl_slider.setMaximum(self.num_frames-1)
        self.ui.brush_size_bar.setMaximum(100)

        self.btn_controller = ButtonController(self.ui, self.cursor)
        self.setup_control()

        #self.resize(self.width, self.height) 
        # initialize
        self.viz_mode = 'davis'
        self.curr_interaction = 'Free'

        self.current_mask = np.zeros((self.num_frames, self.height, self.width), dtype=np.uint8)
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.vis_hist = deque(maxlen=100)
        self.on_showing = None
        self.brush_size = 1
        self.num_objects = 1


        # Zoom parameters
        self.zoom_pixels = 150

        # initialize action
        self.interactions = {}
        self.interactions['interact'] = [[] for _ in range(self.num_frames)]
        self.interactions['annotated_frame'] = []
        self.is_edited = np.zeros((self.num_frames))

        self.this_frame_interactions = []
        self.interaction = "Free"
        self.draw_mode = "draw"
        self.reset_this_interaction()
        self.pressed = False
        self.right_click = False
        self.ctrl_size = False
        self.current_object = 1
        self.last_ex = self.last_ey = 0
        self.ctrl_key = False
        #self.interacted_mask = None
        self.interacted_mask = np.zeros((self.num_objects, self.height, self.width), dtype=np.uint8)
        self.showCurrentFrame()
        self.timestamp_push_text() 

        self.waiting_to_start = True
        self.global_timer = Timer().start()
        self.algo_timer = Timer()
        self.user_timer = Timer()
        self.console_push_text('Initialized.')

    def resizeEvent(self, event):
        self.showCurrentFrame()

    def setup_control(self, ):
        self.ui.tl_slider.valueChanged.connect(self.tl_slide)
        self.ui.brush_size_bar.valueChanged.connect(self.brush_slide)
        self.ui.undo_button.clicked.connect(self.on_undo)
        self.ui.timer.timeout.connect(self.on_time)
        self.ui.reset_button.clicked.connect(self.on_reset)
        self.ui.play_button.clicked.connect(self.on_play)
        self.ui.zoom_m_button.clicked.connect(self.on_zoom_minus)
        self.ui.zoom_p_button.clicked.connect(self.on_zoom_plus)
        self.ui.eraser_button.clicked.connect(self.on_erase)

        #setup the mouse event on main_canvas
        self.ui.main_canvas.mousePressEvent = self.on_press
        self.ui.main_canvas.mouseMoveEvent = self.on_motion
        self.ui.main_canvas.mouseReleaseEvent = self.on_release
        
        # Use to control go next and prev
        QShortcut(QKeySequence(Qt.Key_A), self).activated.connect(self.on_prev)
        QShortcut(QKeySequence(Qt.Key_D), self).activated.connect(self.on_next)
        
        # Use to control play and pause
        QShortcut(QKeySequence('p'), self).activated.connect(self.on_play)

        # Use to control brush_size
        QShortcut(QKeySequence(Qt.Key_1), self).activated.connect(self.on_brsize_minus)
        QShortcut(QKeySequence(Qt.Key_2), self).activated.connect(self.on_brsize_plus)

        QShortcut(QKeySequence(Qt.Key_R), self).activated.connect(self.on_reset)
        QShortcut(QKeySequence(Qt.Key_E), self).activated.connect(self.on_erase)
    
    

    def on_zoom_plus(self):
        self.zoom_pixels -= 25
        self.zoom_pixels = max(50, self.zoom_pixels)
        self.update_minimap()

    def on_zoom_minus(self):
        self.zoom_pixels += 25
        self.zoom_pixels = min(self.zoom_pixels, 300)
        self.update_minimap() 


    def set_navi_enable(self, boolean):
        self.ui.zoom_p_button.setEnabled(boolean)
        self.ui.zoom_m_button.setEnabled(boolean)
        self.ui.infer_button.setEnabled(boolean)
        self.ui.tl_slider.setEnabled(boolean)
        self.ui.play_button.setEnabled(boolean)
        self.ui.frame_log.setEnabled(boolean) 
            
    def on_time(self):
        self.cursor += 1
        if self.cursor > self.num_frames-1:
            self.cursor = 0
        self.ui.tl_slider.setValue(self.cursor)

    def on_erase(self):
        self.draw_mode = "erase" if self.draw_mode == "draw" else "draw"
        if self.draw_mode == "erase":
            self.console_push_text('Enter erase mode.')
        else:
            self.console_push_text('Enter draw mode.')
            
    def on_reset(self):
        # DO not edit prob -- we still need the mask diff
      
        self.current_mask[self.cursor] = load_mask(self.files_path, self.cursor)
        self.reset_this_interaction()
        self.showCurrentFrame()

    def on_play(self):
        if self.ui.timer.isActive():
            self.ui.timer.stop()
        else:
            self.ui.timer.start(1000 / 25)

    def on_prev(self):
        # self.tl_slide will trigger on setValue
        self.cursor = max(2, self.cursor-1)
        self.ui.tl_slider.setValue(self.cursor)

    def on_next(self):
        # self.tl_slide will trigger on setValue
        self.cursor = min(self.cursor+1, self.num_frames-1)
        self.ui.tl_slider.setValue(self.cursor)

    def on_undo(self):

        if self.interaction.can_undo():
            self.interacted_mask = self.interaction.undo()
        else:
            self.reset_this_interaction()
                

        
        self.update_interacted_mask()

    def on_brsize_plus(self):
        self.brush_size += 1
        self.brush_size = min(self.brush_size, self.ui.brush_size_bar.maximum())
        self.ui.brush_size_bar.setValue(self.brush_size)
        self.brush_slide()
        self.clear_brush()
        self.vis_brush(self.last_ex, self.last_ey)
        self.update_interact_vis()
        self.update_minimap()

    def on_brsize_minus(self):
        self.brush_size -= 1
        self.brush_size = max(self.brush_size, 1)
        self.ui.brush_size_bar.setValue(self.brush_size)
        self.brush_slide()
        self.clear_brush()
        self.vis_brush(self.last_ex, self.last_ey)
        self.update_interact_vis()
        self.update_minimap()
        
    def console_push_text(self, text):
        text = '[A: %s, U: %s]: %s' % (self.algo_timer.format(), self.user_timer.format(), text)
        self.ui.console.appendPlainText(text)
        self.ui.console.moveCursor(QTextCursor.End)

    def timestamp_push_text(self):
        self.ui.ts_log.setText(str(self.radar_timestamps[self.cursor]))


    def brush_slide(self):
        self.brush_size = self.ui.brush_size_bar.value()
        self.ui.brush_size_label.setText('Brush size: %d' % self.brush_size)
        try:
            if type(self.interaction) == FreeInteraction:
                self.interaction.set_size(self.brush_size)
        except AttributeError:
            # Initialization, forget about it
            pass
        #self.interaction.set_size(self.brush_size)
    
    

    def tl_slide(self):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            self.console_push_text('Timers started.')

        self.reset_this_interaction()
        self.cursor = self.ui.tl_slider.value()
    
        self.showCurrentFrame()
        self.console_push_text(self.files_path + str(self.cursor) + '.npy')
        self.timestamp_push_text()

    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)


    def vis_brush(self, ex, ey):
       
        self.brush_vis_map = cv2.circle(self.brush_vis_map, 
                (int(round(ex)), int(round(ey))), self.brush_size//2+1, [255, 0, 0], thickness=-1)
        self.brush_vis_alpha = cv2.circle(self.brush_vis_alpha, 
                (int(round(ex)), int(round(ey))), self.brush_size//2+1, 0.5, thickness=-1) 


    def compose_current_im(self):
        
        self.image = load_image(self.files_path, self.cursor)
        if not self.is_edited[self.cursor]:
            # self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))

            self.current_mask[self.cursor] = load_mask(self.files_path, self.cursor)
        self.viz = overlay_moving_mask(self.image, self.current_mask[self.cursor])


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

        ## use to display original image and mask
        self.viz_with_stroke = self.viz*(1-vis_alpha) + vis_map*vis_alpha 

        ## use to display brush
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
        """
        Transform the mouse position in PyQt's main canvas
        to the actual pixel position of image
        """
        
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

    def clear_visualization(self):

        self.vis_map.fill(0)
        self.vis_alpha.fill(0)
        self.vis_hist.clear()
        self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy())) 


    def complete_interaction(self):
        if self.interaction is not None:
            self.clear_visualization()
            self.interactions['annotated_frame'].append(self.cursor)
            self.interactions['interact'][self.cursor].append(self.interaction)
            self.this_frame_interactions.append(self.interaction)
            self.interaction = None

            # reload the saved mask
            self.interacted_mask = np.zeros((self.num_objects, self.height, self.width), dtype=np.uint8)
            self.interacted_mask[0] = load_mask(self.files_path, self.cursor)
            self.ui.undo_button.setDisabled(False)

    def reset_this_interaction(self):
        self.complete_interaction()
        self.clear_visualization()

        self.interaction = None
        self.this_frame_interactions = []
        self.ui.undo_button.setDisabled(True)
    
    def update_interacted_mask(self):
 
        #self.processor.update_mask_only(self.interacted_mask, self.cursur)
        self.current_mask[self.cursor] = self.interacted_mask[0]
        self.is_edited[self.cursor] = True
        self.showCurrentFrame()


    def on_release(self, event):
        self.user_timer.pause()
        ex, ey = self.get_scaled_pos(event.x(), event.y())

        self.console_push_text('Interaction %s at frame %d.' % (self.curr_interaction, self.cursor))
        # Ordinary interaction (might be in local mode)
       
        interaction = self.interaction

        if self.curr_interaction == 'Free':
            self.on_motion(event)
            interaction.end_path()
            if self.curr_interaction == 'Free':
                self.clear_visualization()

        self.interacted_mask = interaction.update()
        self.update_interacted_mask()
        self.pressed = self.ctrl_key = self.right_click = False
        self.ui.undo_button.setDisabled(False)
        self.user_timer.start()

    def on_motion(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.last_ex, self.last_ey = ex, ey
        self.clear_brush()
        # Visualize
        self.vis_brush(ex, ey)
        if self.pressed:
            if not self.ctrl_key:
                if self.curr_interaction == 'Free':
                    if self.draw_mode == 'draw':
                        obj = 0 if self.right_click else self.current_object
                        # Actually draw it if dragging
                        self.vis_map, self.vis_alpha = self.interaction.push_point(
                            ex, ey, obj, (self.vis_map, self.vis_alpha), mode="draw"
                        )
                    elif self.draw_mode == 'erase':
                        self.vis_map, self.vis_alpha = self.interaction.push_point(
                            ex, ey, 0, (self.vis_map, self.vis_alpha), mode="erase"
                        )
                
                        
        self.update_interact_vis()
        self.update_minimap()

    def on_press(self, event):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            self.console_push_text('Timers started.')
        self.user_timer.pause()
        self.ctrl_key = False

        self.pressed = True
        self.right_click = (event.button() != 1)
        self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))
        
        last_interaction = self.interaction
        
        new_interaction = None
        if self.curr_interaction == 'Free':
            if last_interaction is None or type(last_interaction) != FreeInteraction:
                self.complete_interaction()
                self.mask = load_mask(self.files_path, self.cursor)
                new_interaction = FreeInteraction(self.interacted_mask, self.mask, 
                            self.num_objects)
                new_interaction.set_size(self.brush_size)
        if new_interaction is not None:
                self.interaction = new_interaction

        # Just motion it as the first step
        self.on_motion(event)
        self.user_timer.start()
        
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.on_zoom_plus()
        else:
            self.on_zoom_minus()
 
if __name__ == "__main__":
    pass
