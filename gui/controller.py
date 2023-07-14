import numpy as np
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QKeySequence, QImage, QTextCursor, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QSizePolicy,
    QButtonGroup,
    QSlider,
    QShortcut,
    QRadioButton,
    QProgressBar,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer

from icecream import ic
from .ui import Ui_main_widget
from .interact.timer import Timer
from .interact.interaction import *
from .utils.gui_utils import overlay_moving_mask
from .utils.file_utils import *
from .utils.image_method import ImageMethod
from .btn_controller import ButtonController
from .inference_core import Inference_core
from .shortcut import Shortcut
import ipdb
import time


class MainWindow_controller(QtWidgets.QWidget):
    def __init__(self, files_path, timestamps, sam_controller, patch_num):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx

        # Set up some initial values
        self.cursor = 5  # represent the current frame index
        self.radar_timestamps = timestamps
        self.files_path = files_path
        self.num_frames = len(self.radar_timestamps)
        if patch_num != -1:
            self.dataloader = Patch_DataLoader(
                self.files_path, self.radar_timestamps, patch_num
            )
        else:
            self.dataloader = DataLoader(self.files_path, self.radar_timestamps)
        self.dataloader.load_data(self.cursor)
        self.image = self.dataloader.load_image()
        self.mask = self.dataloader.load_mask()
        # self.lidar_mask = self.dataloader.load_lidar_mask()
        # self.camera_image = self.dataloader.load_camera()
        self.width, self.height = self.image.shape[:2]
        self.Buttoncontroller = ButtonController(self)
        self.Shortcut = Shortcut(self, self.Buttoncontroller)
        self.sam_controller = sam_controller
        self.processor = Inference_core()
        self.image_method = ImageMethod(self)

        self.ui = Ui_main_widget()
        self.ui.setupUi(self)
        self.setLayout(self.ui.layout)
        self.setWindowTitle("SR Label")
        self.setGeometry(1080, 100, 1920, 1080)

        # setup the slider tick and minimum accroding to the train/test split
        # for train the tick = 2, for test tick = 1
        self.ui.tl_slider.setSingleStep(1)
        self.ui.tl_slider.setMinimum(self.cursor)
        self.ui.tl_slider.setMaximum(self.num_frames - 1)
        self.ui.brush_size_bar.setMaximum(40)

        self.setup_control()

        # initialize

        self.current_mask = np.zeros(
            (self.num_frames, self.height, self.width), dtype=np.uint8
        )
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.lidar_mask_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.vis_hist = deque(maxlen=100)
        self.on_showing = None
        self.brush_size = 1
        self.num_objects = 1
        self.brightness = 20
        self.contrast = 20
        self.threshold = 20
        self.current_brightness = 0
        # Zoom parameters
        self.zoom_pixels = 150

        # initialize action
        self.curr_interaction = "Free"

        self.interactions = {}
        self.interactions["interact"] = [[] for _ in range(self.num_frames)]
        self.interactions["annotated_frame"] = []
        self.is_edited = np.zeros((self.num_frames))

        self.this_frame_interactions = []
        self.interaction = "Free"
        self.draw_mode = "draw"

        self.current_patch = patch_num
        self.pressed = False
        self.right_click = False
        self.left_click = False
        self.ctrl_size = False
        self.current_object = 1
        self.last_ex = self.last_ey = 0
        self.brush_step = 2
        self.ctrl_key = False

        self.embedding_created = False
        self.auto_save_mode = False
        self.next_flag = False
        self.prev_flag = False
        self.play_flag = False
        self.is_saved_flag = False
        self.thres_mode = False
        self.mask_mode = True
        self.lidar_mask_mode = True
        self.interacted_mask = np.zeros(
            (self.num_objects, self.height, self.width), dtype=np.uint8
        )

        self.ui.brightness_bar.setValue(self.brightness)

        self.reset_this_interaction()
        self.showCurrentFrame()
        self.timestamp_push_text()

        self.waiting_to_start = True
        self.global_timer = Timer().start()
        self.algo_timer = Timer()
        self.user_timer = Timer()
        self.console_push_text("Initialized.")
        self.sam_controller.set_image(self.image)

    def resizeEvent(self, event):
        self.showCurrentFrame()

    def setup_control(
        self,
    ):
        """
        Binding the trigger function to the button or slide bar
        """
        self.ui.tl_slider.valueChanged.connect(self.tl_slide)
        self.ui.brush_size_bar.valueChanged.connect(self.brush_slide)
        self.ui.radio_bbox.toggled.connect(self.interaction_radio_clicked)
        self.ui.radio_free.toggled.connect(self.interaction_radio_clicked)

        self.ui.undo_button.clicked.connect(self.Buttoncontroller.on_undo)
        self.ui.timer.timeout.connect(self.Buttoncontroller.on_time)
        self.ui.reset_button.clicked.connect(self.Buttoncontroller.on_reset)
        self.ui.play_button.clicked.connect(self.Buttoncontroller.on_play)
        self.ui.zoom_m_button.clicked.connect(self.Buttoncontroller.on_zoom_minus)
        self.ui.zoom_p_button.clicked.connect(self.Buttoncontroller.on_zoom_plus)
        self.ui.eraser_button.clicked.connect(self.Buttoncontroller.on_erase)
        self.ui.save_button.clicked.connect(self.Buttoncontroller.on_save)
        self.ui.auto_save_btn.stateChanged.connect(self.set_auto_save_mode)
        self.ui.model_button.clicked.connect(self.load_model)
        self.ui.infer_button.clicked.connect(self.Buttoncontroller.on_infer)
        self.ui.brightness_bar.valueChanged.connect(self.brightness_slide)
        self.ui.contrast_bar.valueChanged.connect(self.contrast_slide)
        self.ui.threshold_bar.valueChanged.connect(self.threshold_slide)
        self.ui.infer_button.setEnabled(False)
        self.ui.portion1_button.clicked.connect(
            self.Buttoncontroller.on_switch_to_patch_0
        )
        self.ui.portion2_button.clicked.connect(
            self.Buttoncontroller.on_switch_to_patch_1
        )
        self.ui.portion3_button.clicked.connect(
            self.Buttoncontroller.on_switch_to_patch_2
        )
        self.ui.portion4_button.clicked.connect(
            self.Buttoncontroller.on_switch_to_patch_3
        )

        # setup the mouse event on main_canvas
        self.ui.main_canvas.mousePressEvent = self.on_press
        self.ui.main_canvas.mouseMoveEvent = self.on_motion
        self.ui.main_canvas.mouseReleaseEvent = self.on_release

    def load_model(self):
        # loading the model when the button is clicked
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("PyTorch Model Files (*.pth)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_() == QFileDialog.Accepted:
            self.model_path = file_dialog.selectedFiles()[0]

            if self.model_path is not None:
                try:
                    self.processor.set_model(self.model_path)
                    self.console_push_text(
                        "Loaded pretrained model from {}".format(self.model_path)
                    )
                    self.ui.infer_button.setEnabled(True)

                except FileNotFoundError:
                    self.console_push_text(
                        "Failed to load model... check the model path is correct"
                    )

    def set_auto_save_mode(self, state):
        if state == 0:
            self.auto_save_mode = False
            self.ui.play_button.setDisabled(False)
        else:
            self.auto_save_mode = True
            self.ui.play_button.setDisabled(True)

    def set_navi_disable(self, boolean):
        self.ui.save_button.setDisabled(boolean)
        self.ui.model_button.setDisabled(boolean)
        self.ui.reset_button.setDisabled(boolean)
        self.ui.eraser_button.setDisabled(boolean)

    def set_continue(self):
        if not self.auto_save_mode:
            msg_box = QMessageBox(self)
            msg = self.tr(
                "You have unsaved changes. Do you want to save before proceeding?"
            )
            answer = msg_box.question(
                self,
                self.tr("Save annotations?"),
                msg,
                msg_box.Save | msg_box.Discard | msg_box.Cancel,
                msg_box.Save,
            )
            if answer == msg_box.Save:
                self.Buttoncontroller.on_save()
                return True
            elif answer == msg_box.Discard:
                return True
            elif answer == msg_box.Cancel:
                return False
        else:
            self.Buttoncontroller.on_save()
            return True

    def interaction_radio_clicked(self, event):
        self.last_interaction = self.curr_interaction
        if self.ui.radio_free.isChecked():
            self.ui.brush_size_bar.setDisabled(False)
            self.brush_slide()
            self.curr_interaction = "Free"
        elif self.ui.radio_bbox.isChecked():
            self.curr_interaction = "Box"
            self.brush_size = 1
            self.ui.brush_size_bar.setDisabled(True)

    def console_push_text(self, text):
        text = "[A: %s, U: %s]: %s" % (
            self.algo_timer.format(),
            self.user_timer.format(),
            text,
        )
        self.ui.console.appendPlainText(text)
        self.ui.console.moveCursor(QTextCursor.End)

    def timestamp_push_text(self):
        self.ui.ts_log.setText(str(self.radar_timestamps[self.cursor]))

    def brightness_slide(self):
        self.brightness = self.ui.brightness_bar.value()
        self.showCurrentFrame()

    def threshold_slide(self):
        self.threshold = self.ui.threshold_bar.value()
        self.showCurrentFrame()

    def contrast_slide(self):
        self.contrast = self.ui.contrast_bar.value()
        self.showCurrentFrame()

    def brush_slide(self):
        self.brush_size = self.ui.brush_size_bar.value()
        self.ui.brush_size_label.setText("Brush size: %d" % self.brush_size)
        try:
            if type(self.interaction) == FreeInteraction:
                self.interaction.set_size(self.brush_size)
        except AttributeError:
            pass

    def tl_slide(self):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            self.console_push_text("Timers started.")
        self.is_saved_flag = False
        self.cursor = self.ui.tl_slider.value()

        self.dataloader.load_data(self.cursor)
        if self.dataloader.is_valid():
            self.reset_this_interaction()
            self.showCurrentFrame()
            self.timestamp_push_text()
        else:
            if self.prev_flag:
                self.cursor = self.cursor - 1
                self.ui.tl_slider.setValue(self.cursor)
            elif self.next_flag:
                self.cursor = self.cursor + 1
                self.ui.tl_slider.setValue(self.cursor)
        self.prev_flag = False
        self.next_flag = False
        self.embedding_created = False

    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)

    def vis_brush(self, ex, ey):
        self.brush_vis_map = cv2.circle(
            self.brush_vis_map,
            (int(round(ex)), int(round(ey))),
            self.brush_size // 2 + 1,
            [255, 0, 0],
            thickness=-1,
        )
        self.brush_vis_alpha = cv2.circle(
            self.brush_vis_alpha,
            (int(round(ex)), int(round(ey))),
            self.brush_size // 2 + 1,
            0.5,
            thickness=-1,
        )

    def compose_current_im(self):
        self.image = self.dataloader.load_image()
        self.camera_image = self.dataloader.load_camera()

        # display mode switch
        if self.thres_mode:
            self.image = self.image_method.image_to_threshold(self.image)
        elif self.lidar_mask_mode and self.current_patch != -1:
            self.lidar_mask, self.lidar_mask_alpha = self.dataloader.load_lidar_mask()

        self.image = self.image_method.image_to_brightness(self.image)
        self.image = self.image_method.image_to_contrast(self.image)
        if not self.is_edited[self.cursor]:
            self.current_mask[self.cursor] = self.dataloader.load_mask()
        self.viz = overlay_moving_mask(self.image, self.current_mask[self.cursor])

    def update_minimap(self):
        # Limit it within the valid range
        ex, ey = self.last_ex, self.last_ey
        r = self.zoom_pixels // 2
        ex = int(round(max(r, min(self.width - r, ex))))
        ey = int(round(max(r, min(self.height - r, ey))))

        patch = self.viz_with_stroke[ey - r : ey + r, ex - r : ex + r, :].astype(
            np.uint8
        )
        if self.current_patch != -1:
            patch = self.camera_image
        height, width, channel = patch.shape
        bytesPerLine = 3 * width
        qImg = QImage(patch.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.ui.minimap.setPixmap(
            QPixmap(
                qImg.scaled(
                    self.ui.minimap.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )
        )

    def update_interact_vis(self):
        # Update the interactions without re-computing the overlay
        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        vis_map = self.vis_map
        vis_alpha = self.vis_alpha
        brush_vis_map = self.brush_vis_map
        brush_vis_alpha = self.brush_vis_alpha
        if self.current_patch != -1:
            lidar_mask_map = self.lidar_mask
            lidar_mask_alpha = self.lidar_mask_alpha
            ## use to display original image and mask

        self.viz_with_stroke = self.viz * (1 - vis_alpha) + vis_map * vis_alpha
        if self.lidar_mask_mode and self.current_patch != -1:
            self.viz_with_stroke = (
                self.viz_with_stroke * (1 - lidar_mask_alpha)
                + lidar_mask_map * lidar_mask_alpha
            )
        ## use to display brush
        self.viz_with_stroke = (
            self.viz_with_stroke * (1 - brush_vis_alpha)
            + brush_vis_map * brush_vis_alpha
        )
        self.viz_with_stroke = self.viz_with_stroke.astype(np.uint8)

        qImg = QImage(
            self.viz_with_stroke.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        self.ui.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(
                    self.ui.main_canvas.size(),
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation,
                )
            )
        )

        self.main_canvas_size = self.ui.main_canvas.size()
        self.image_size = qImg.size()

    def showCurrentFrame(self):
        self.compose_current_im()
        self.update_interact_vis()
        self.update_minimap()
        self.ui.frame_log.setText(f"{self.cursor}/ {self.num_frames-1}")
        self.ui.tl_slider.setValue(self.cursor)

    def setBrushSize(self):
        self.ui.brush_size_label.setText(
            f"Brush Size: {self.ui.brush_size_bar.value()}"
        )

    def get_scaled_pos(self, x, y):
        """
        Transform the mouse position in PyQt's main canvas
        to the actual pixel position of image
        """

        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x -= (fw - ow) / 2
        y -= (fh - oh) / 2

        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
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
            self.interactions["annotated_frame"].append(self.cursor)
            self.interactions["interact"][self.cursor].append(self.interaction)
            self.this_frame_interactions.append(self.interaction)
            self.interaction = None
            self.ui.undo_button.setDisabled(False)
        else:
            self.interacted_mask = np.zeros(
                (self.num_objects, self.height, self.width), dtype=np.uint8
            )
            self.interacted_mask[0] = self.dataloader.load_mask()
            self.ui.undo_button.setDisabled(False)

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_mask[0] = self.dataloader.load_mask()
        self.update_interacted_mask()
        self.clear_visualization()
        self.interaction = None
        self.this_frame_interactions = []
        self.ui.undo_button.setDisabled(True)

    def update_interacted_mask(self):
        # self.processor.update_mask_only(self.interacted_mask, self.cursur)
        self.current_mask[self.cursor] = self.interacted_mask[0]
        self.is_edited[self.cursor] = True
        self.showCurrentFrame()

    def on_release(self, event):
        self.user_timer.pause()
        ex, ey = self.get_scaled_pos(event.x(), event.y())

        self.console_push_text(
            "Interaction %s at frame %d." % (self.curr_interaction, self.cursor)
        )
        # Ordinary interaction (might be in local mode)

        interaction = self.interaction

        if self.curr_interaction == "Free":
            self.on_motion(event)
            interaction.end_path()
            # reset the brush layer
            self.clear_visualization()
            self.interacted_mask = interaction.update()
            self.update_interacted_mask()
            self.right_click = self.left_click = False
            self.ui.undo_button.setDisabled(False)
        elif self.curr_interaction == "Box":
            if self.right_click:
                self.right_click = False
                # self.ui.undo_button.setDisabled(True)
            elif self.left_click:
                self.on_motion(event)
                interaction.end_path()
                self.clear_visualization()
                self.interacted_mask = interaction.update()
                self.update_interacted_mask()
                self.left_click = False
                self.ui.undo_button.setDisabled(False)

        self.pressed = self.ctrl_key = False

        self.user_timer.start()

    def on_motion(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.last_ex, self.last_ey = ex, ey
        self.clear_brush()
        # Visualize
        self.vis_brush(ex, ey)

        if self.pressed:
            if not (self.left_click and self.right_click):
                if self.curr_interaction == "Free":
                    if self.draw_mode == "draw":
                        obj = 0 if self.right_click else self.current_object
                        # Actually draw it if dragging
                        self.vis_map, self.vis_alpha = self.interaction.push_point(
                            ex, ey, obj, (self.vis_map, self.vis_alpha), mode="draw"
                        )

                    elif self.draw_mode == "erase":
                        self.vis_map, self.vis_alpha = self.interaction.push_point(
                            ex, ey, 0, (self.vis_map, self.vis_alpha), mode="erase"
                        )
                elif self.curr_interaction == "Box":
                    self.clear_visualization()

                    if self.draw_mode == "draw" and self.left_click:
                        obj = self.current_object
                        self.vis_map, self.vis_alpha = self.interaction.push_point(
                            ex, ey, obj, (self.vis_map, self.vis_alpha), mode="draw"
                        )

        self.update_interact_vis()
        self.update_minimap()

    def on_press(self, event):
        if self.waiting_to_start:
            self.waiting_to_start = False
            self.algo_timer.start()
            self.user_timer.start()
            self.console_push_text("Timers started.")
        self.user_timer.pause()
        self.ctrl_key = False

        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.pressed = True

        # Deal with pressing left and right click at the same time
        if self.right_click:
            self.left_click = event.button() == 1
        elif self.left_click:
            self.right_click = event.button() == 2
        elif not (self.right_click and self.left_click):
            self.right_click = event.button() == 2
            self.left_click = event.button() == 1

        self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))
        last_interaction = self.interaction

        new_interaction = None
        if self.curr_interaction == "Free":
            if last_interaction is None or type(last_interaction) != FreeInteraction:
                self.complete_interaction()
                self.mask = self.dataloader.load_mask()
                new_interaction = FreeInteraction(
                    self.interacted_mask, self.num_objects, self.processor
                )
                new_interaction.set_size(self.brush_size)
        elif self.curr_interaction == "Box":
            if last_interaction is None or type(last_interaction) != BoxInteraction:
                if self.left_click:
                    self.complete_interaction()
                self.mask = self.dataloader.load_mask()
                if not self.embedding_created:
                    self.sam_controller.set_image(self.dataloader.load_image())
                    self.embedding_created = True
                new_interaction = BoxInteraction(
                    self.interacted_mask, self.num_objects, self.sam_controller
                )
                new_interaction.set_init_pos(ex, ey)
                new_interaction.set_size(self.brush_size)
            else:
                last_interaction.set_init_pos(ex, ey)
        if new_interaction is not None:
            self.interaction = new_interaction
        # Just motion it as the first step
        self.on_motion(event)
        self.user_timer.start()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.Buttoncontroller.on_zoom_plus()
        else:
            self.Buttoncontroller.on_zoom_minus()


if __name__ == "__main__":
    pass
