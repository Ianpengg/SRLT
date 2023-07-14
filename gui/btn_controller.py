# from .utils.file_utils import load_image, load_mask, load_data, save_mask

from .interact.interaction import FreeInteraction
import numpy as np


class ButtonController:
    def __init__(self, controller):
        self.controller = controller

    def on_save(self):
        self.controller.is_saved_flag = True
        self.controller.dataloader.save_mask(self.controller.interacted_mask[0])
        self.controller.console_push_text(f"Annotated mask saved.")

    def on_time(self):
        self.controller.cursor += 1
        if self.controller.cursor > self.controller.num_frames - 1:
            self.controller.cursor = 0
        self.controller.ui.tl_slider.setValue(self.controller.cursor)

    def on_erase(self):
        self.controller.draw_mode = (
            "erase" if self.controller.draw_mode == "draw" else "draw"
        )
        if self.controller.draw_mode == "erase":
            self.controller.ui.eraser_button.setStyleSheet("background-color: red")
            self.controller.console_push_text("Enter erase mode.")
        else:
            self.controller.ui.eraser_button.setStyleSheet("background-color: None")
            self.controller.console_push_text("Enter draw mode.")

    def on_reset(self):
        # DO not edit prob -- we still need the mask diff

        self.controller.current_mask[
            self.controller.cursor
        ] = self.controller.dataloader.load_mask()
        self.controller.reset_this_interaction()
        self.controller.showCurrentFrame()

    def on_play(self):
        self.controller.play_flag = (
            True if self.controller.play_flag == False else False
        )
        self.controller.ui.play_button.setStyleSheet(
            "background-color: red"
            if self.controller.play_flag
            else "background-color: None"
        )
        self.controller.set_navi_disable(self.controller.play_flag)
        if self.controller.ui.timer.isActive():
            self.controller.ui.timer.stop()
        else:
            self.controller.ui.timer.start(1000 / 25)

    def on_prev(self):
        if not self.controller.pressed:
            self.controller.prev_flag = True
            if self.controller.is_saved_flag:
                self.controller.cursor = max(5, self.controller.cursor - 1)
                self.controller.ui.tl_slider.setValue(self.controller.cursor)
            elif not self.controller.is_saved_flag and self.controller.set_continue():
                self.controller.cursor = max(5, self.controller.cursor - 1)
                self.controller.ui.tl_slider.setValue(self.controller.cursor)

    def on_next(self):
        if not self.controller.pressed:
            self.controller.next_flag = True
            if self.controller.is_saved_flag:
                self.controller.cursor = min(
                    self.controller.cursor + 1, self.controller.num_frames - 1
                )
                self.controller.ui.tl_slider.setValue(self.controller.cursor)
            elif not self.controller.is_saved_flag and self.controller.set_continue():
                self.controller.cursor = min(
                    self.controller.cursor + 1, self.controller.num_frames - 1
                )
                self.controller.ui.tl_slider.setValue(self.controller.cursor)

    def on_undo(self):
        if self.controller.interaction is not None:
            if self.controller.interaction.can_undo():
                self.controller.interacted_mask = self.controller.interaction.undo()
            else:
                if len(self.controller.this_frame_interactions) > 0:
                    self.controller.interacted_mask = self.controller.interaction.undo()
                    self.controller.interaction = (
                        self.controller.this_frame_interactions[-1]
                    )
                    _ = self.controller.this_frame_interactions.pop()

                else:
                    self.controller.interacted_mask = self.controller.interaction.undo()
                    self.controller.reset_this_interaction()
        else:
            self.controller.reset_this_interaction()
        self.controller.update_interacted_mask()

    def on_infer(self):
        if self.controller.processor.model is not None:
            # infer the current frame
            data = self.controller.dataloader.load_data()
            # if there is no interaction, create a new one
            if self.controller.interaction is None:
                self.controller.interaction = FreeInteraction(
                    self.controller.interacted_mask,
                    self.controller.mask,
                    self.controller.num_objects,
                    self.controller.processor,
                )
                self.controller.interacted_mask[
                    0
                ] = self.controller.interaction.predict(data)
                self.controller.ui.undo_button.setDisabled(False)

            else:
                self.controller.interacted_mask[
                    0
                ] = self.controller.interaction.predict(data)

            self.controller.update_interacted_mask()

    def on_zoom_plus(self):
        self.controller.zoom_pixels -= 25
        self.controller.zoom_pixels = max(50, self.controller.zoom_pixels)
        self.controller.update_minimap()

    def on_zoom_minus(self):
        self.controller.zoom_pixels += 25
        self.controller.zoom_pixels = min(self.controller.zoom_pixels, 300)
        self.controller.update_minimap()

    def on_brsize_plus(self):
        self.controller.brush_size += self.controller.brush_step
        self.controller.brush_size = min(
            self.controller.brush_size, self.controller.ui.brush_size_bar.maximum()
        )
        self.controller.ui.brush_size_bar.setValue(self.controller.brush_size)
        self.controller.brush_slide()
        self.controller.clear_brush()
        self.controller.vis_brush(self.controller.last_ex, self.controller.last_ey)
        self.controller.update_interact_vis()
        self.controller.update_minimap()

    def on_brsize_minus(self):
        self.controller.brush_size -= self.controller.brush_step
        self.controller.brush_size = max(self.controller.brush_size, 1)
        self.controller.ui.brush_size_bar.setValue(self.controller.brush_size)
        self.controller.brush_slide()
        self.controller.clear_brush()
        self.controller.vis_brush(self.controller.last_ex, self.controller.last_ey)
        self.controller.update_interact_vis()
        self.controller.update_minimap()

    def on_switch_mask(self):
        # set brightness to zero to show the mask
        # set brightness back to current value to see original image

        # save current brightness

        if self.controller.mask_mode:
            self.controller.current_brightness = (
                self.controller.ui.brightness_bar.value()
            )
            self.controller.ui.brightness_bar.setValue(0)
            self.controller.mask_mode = False
            self.controller.showCurrentFrame()
        else:
            self.controller.ui.brightness_bar.setValue(
                self.controller.current_brightness
            )
            self.controller.mask_mode = True
            self.controller.showCurrentFrame()

    def on_switch_lidar_mask(self):
        # set brightness to zero to show the mask
        # set brightness back to current value to see original image

        # save current brightness

        if self.controller.lidar_mask_mode:
            self.controller.lidar_mask_mode = False
            self.controller.showCurrentFrame()
        else:
            self.controller.lidar_mask_mode = True
            self.controller.showCurrentFrame()

    def on_switch_threshold(self):
        if not self.controller.thres_mode:
            self.controller.ui.threshold_bar.setValue(self.controller.threshold)
            self.controller.thres_mode = True
            self.controller.showCurrentFrame()
        else:
            self.controller.thres_mode = False
            self.controller.showCurrentFrame()

    def on_switch_to_free(self):
        self.controller.curr_interaction = "Free"
        self.controller.ui.radio_free.toggle()

    def on_switch_to_box(self):
        self.controller.curr_interaction = "Box"
        self.controller.ui.radio_bbox.toggle()

    def on_switch_to_patch_0(self):
        self.controller.dataloader.set_patch_index(np.array([0, 0]))
        self.controller.dataloader.load_data(self.controller.cursor)
        self.controller.reset_this_interaction()
        self.controller.showCurrentFrame()

    def on_switch_to_patch_1(self):
        self.controller.dataloader.set_patch_index(np.array([1, 1]))
        self.controller.dataloader.load_data(self.controller.cursor)
        self.controller.reset_this_interaction()
        self.controller.showCurrentFrame()

    def on_switch_to_patch_2(self):
        self.controller.dataloader.set_patch_index(np.array([2, 2]))
        self.controller.dataloader.load_data(self.controller.cursor)
        self.controller.reset_this_interaction()
        self.controller.showCurrentFrame()

    def on_switch_to_patch_3(self):
        self.controller.dataloader.set_patch_index(np.array([3, 3]))
        self.controller.dataloader.load_data(self.controller.cursor)
        self.controller.reset_this_interaction()
        self.controller.showCurrentFrame()
