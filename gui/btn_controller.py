from .utils.file_utils import load_image, load_mask, load_data, save_mask

from .interact.interaction import FreeInteraction



class ButtonController:
    def __init__(self, controller):
        self.controller = controller


    def on_save(self):
        self.controller.is_saved_flag = True
        save_mask(self.controller.files_path, self.controller.cursor, self.controller.interacted_mask[0])    
        self.controller.console_push_text(f'{self.controller.files_path + str(self.controller.cursor) }.npy Saved.')   
            
    def on_time(self):
        self.controller.cursor += 1
        if self.controller.cursor > self.controller.num_frames-1:
            self.controller.cursor = 0
        self.controller.ui.tl_slider.setValue(self.controller.cursor)

    def on_erase(self):
        self.controller.draw_mode = "erase" if self.controller.draw_mode == "draw" else "draw"
        if self.controller.draw_mode == "erase":
            self.controller.ui.eraser_button.setStyleSheet('background-color: red')
            self.controller.console_push_text('Enter erase mode.')
        else:
            self.controller.ui.eraser_button.setStyleSheet('background-color: None')
            self.controller.console_push_text('Enter draw mode.')
            
    def on_reset(self):
        # DO not edit prob -- we still need the mask diff
      
        self.controller.current_mask[self.controller.cursor] = load_mask(self.controller.files_path, self.controller.cursor)
        self.controller.reset_this_interaction()
        self.controller.showCurrentFrame()
    
    def on_play(self):
        self.controller.play_flag = True if self.controller.play_flag == False else False
        self.controller.ui.play_button.setStyleSheet('background-color: red' if self.controller.play_flag else 'background-color: None')
        self.controller.set_navi_disable(self.controller.play_flag)
        if self.controller.ui.timer.isActive():
            self.controller.ui.timer.stop()
        else:
            self.controller.ui.timer.start(1500 / 25)

    def on_prev(self):
        self.controller.prev_flag = True 
        if self.controller.is_saved_flag:
            self.controller.cursor = max(2, self.controller.cursor-1)
            self.controller.ui.tl_slider.setValue(self.controller.cursor)
        elif not self.controller.is_saved_flag and self.controller.set_continue():
            self.controller.cursor = max(2, self.controller.cursor-1)
            self.controller.ui.tl_slider.setValue(self.controller.cursor)
            
    def on_next(self): 
        self.controller.next_flag = True
        if self.controller.is_saved_flag:
            self.controller.cursor = min(self.controller.cursor+1, self.controller.num_frames-1)
            self.controller.ui.tl_slider.setValue(self.controller.cursor)
        elif not self.controller.is_saved_flag and self.controller.set_continue()  :
            self.controller.cursor = min(self.controller.cursor+1, self.controller.num_frames-1)
            self.controller.ui.tl_slider.setValue(self.controller.cursor)
 

    def on_undo(self):
        if self.controller.interaction is not None:
            if self.controller.interaction.can_undo():
                self.controller.interacted_mask = self.controller.interaction.undo()
            else:
                self.controller.reset_this_interaction() 
        else:
            self.controller.reset_this_interaction()
        self.controller.update_interacted_mask()

    def on_infer(self):
        if self.controller.processor.model is not None:
            # infer the current frame
            data = load_data(self.controller.files_path, self.controller.cursor)
            # if there is no interaction, create a new one
            if self.controller.interaction is None:
                self.controller.interaction = FreeInteraction(self.controller.interacted_mask, self.controller.mask, 
                            self.controller.num_objects, self.controller.processor)
                self.controller.interacted_mask[0] = self.controller.interaction.predict(data)
                self.controller.ui.undo_button.setDisabled(False)
                
            else :
                self.controller.interacted_mask[0] = self.controller.interaction.predict(data)

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
        self.controller.brush_size = min(self.controller.brush_size, self.controller.ui.brush_size_bar.maximum())
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



    