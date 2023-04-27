from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt


class Shortcut:
    """
    Use this class to control the shortcut of the program
    You can easily add new shortcut by adding new function in the Buttoncontroller class
    and add the shortcut in the __init__ function
    """
    def __init__(self, MainWidget,  Buttoncontroller):
        self.Buttoncontroller = Buttoncontroller
        self.MainWidget = MainWidget

        # Use to control go prev
        QShortcut(QKeySequence(Qt.Key_A), self.MainWidget).activated.connect(self.Buttoncontroller.on_prev)
        
        # Use to control go next
        QShortcut(QKeySequence(Qt.Key_D), self.MainWidget).activated.connect(self.Buttoncontroller.on_next)

        # Use to control play and pause
        QShortcut(QKeySequence('p'), self.MainWidget).activated.connect(self.Buttoncontroller.on_play)

        # Use to reduce brush_size
        QShortcut(QKeySequence(Qt.Key_1), self.MainWidget).activated.connect(self.Buttoncontroller.on_brsize_minus)

        # Use to increase brush_size
        QShortcut(QKeySequence(Qt.Key_2), self.MainWidget).activated.connect(self.Buttoncontroller.on_brsize_plus)

        # Use to control reset
        QShortcut(QKeySequence(Qt.Key_R), self.MainWidget).activated.connect(self.Buttoncontroller.on_reset)

        # Use to control zoom
        QShortcut(QKeySequence(Qt.Key_E), self.MainWidget).activated.connect(self.Buttoncontroller.on_erase)

        # Use to control save
        QShortcut(QKeySequence('Ctrl+S'), self.MainWidget).activated.connect(self.Buttoncontroller.on_save)

        # Use to control undo
        QShortcut(QKeySequence('Ctrl+Z'), self.MainWidget).activated.connect(self.Buttoncontroller.on_undo)

        # Use to control infer
        QShortcut(QKeySequence(Qt.Key_I), self.MainWidget).activated.connect(self.Buttoncontroller.on_infer)   
        
        # Use to switch between mask and current frame
        QShortcut(QKeySequence(Qt.Key_W), self.MainWidget).activated.connect(self.Buttoncontroller.on_switch_mask)

        #Use to switch between threshold image and current frame
        QShortcut(QKeySequence(Qt.Key_Q), self.MainWidget).activated.connect(self.Buttoncontroller.on_switch_threshold)
