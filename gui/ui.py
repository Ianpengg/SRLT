# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widget.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer 
from PyQt5.QtWidgets import (QWidget, QApplication, QComboBox, 
    QHBoxLayout, QLabel, QPushButton, QTextEdit, 
    QPlainTextEdit, QVBoxLayout, QSizePolicy, QButtonGroup, QSlider, 
    QShortcut, QRadioButton, QProgressBar, QFileDialog)
class Ui_main_widget(object):
    def setupUi(self, main_widget):
        #main_widget.setObjectName("SR Label")
        #main_widget.resize(1518, 869)
        # self.width = width
        # self.height = height
        # print(self.width)

        # Some buttons
        self.play_button = QPushButton("Play")
        self.undo_button = QPushButton("Undo")
        self.reset_button = QPushButton("Reset")

        self.model_button = QPushButton("Model")
        self.infer_button = QPushButton("Inference")
        self.save_button = QPushButton("Save")

        # Display the frame count 
        self.frame_log = QTextEdit()
        self.frame_log.setMaximumHeight(28)
        self.frame_log.setMaximumWidth(120)
        self.frame_log.setReadOnly(True)

        # display the timeline slider
        self.tl_slider = QSlider(Qt.Horizontal)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TicksBelow)
        self.tl_slider.setTickInterval(1)

        #Brush size slider
                
        self.brush_size_bar = QSlider(Qt.Horizontal)
        self.brush_size_bar.setMinimumWidth(300)
        self.brush_size_bar.setMinimum(1)
        self.brush_size_bar.setMaximum(4)
        self.brush_size_bar.setValue(1)
        self.brush_size_bar.setTickPosition(QSlider.TicksBelow)
        self.brush_size_bar.setTickInterval(1)


        self.brush_size_label = QLabel()
        self.brush_size_label.setMinimumWidth(100)
        self.brush_size_label.setAlignment(Qt.AlignCenter)


         # Main canvas -> QLabel

        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.main_canvas.setMinimumSize(100, 100)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        self.main_canvas.setMouseTracking(True)

        # Minimap -> Also a QLbal
        self.minimap = QLabel()
        self.minimap.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.minimap.setMinimumSize(100, 100)
        self.minimap.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Zoom-in buttons
        self.zoom_p_button = QPushButton('Zoom +')

        self.zoom_m_button = QPushButton('Zoom -')

        # Record the current timestamp information

        self.ts_label = QLabel("Timestamp:")
        self.ts_label.setMaximumSize(QtCore.QSize(100, 40))
        self.ts_label.setAlignment(Qt.AlignCenter)
        self.ts_log = QtWidgets.QTextEdit()
        self.ts_log.setMinimumSize(QtCore.QSize(300, 40 ))
        self.ts_log.setMaximumHeight(40)
        self.ts_log.setReadOnly(True)

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setMinimumSize(QtCore.QSize(100, 100))
        
        self.console.setReadOnly(True)

        # Progress bar
        self.progress = QProgressBar(main_widget)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMinimumWidth(300)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setFormat('Idle')
        self.progress.setStyleSheet("QProgressBar{color: black;}")
        self.progress.setAlignment(Qt.AlignCenter)

        # navigator
        self.navi = QHBoxLayout()
        

        self.navi.addWidget(self.frame_log)
        self.navi.addWidget(self.play_button)

        # Add interac_box
        self.interact_subbox = QtWidgets.QVBoxLayout()
        self.interact_topbox = QtWidgets.QHBoxLayout()
        self.interact_botbox = QtWidgets.QHBoxLayout()

        self.interact_topbox.addWidget(self.brush_size_label)
        self.interact_botbox.addWidget(self.brush_size_bar)
        self.interact_subbox.addLayout(self.interact_topbox)
        self.interact_subbox.addLayout(self.interact_botbox)

        self.navi.addStretch(1)
        self.navi.addWidget(self.undo_button)
        self.navi.addWidget(self.reset_button)

        self.navi.addStretch(1)
        self.navi.addWidget(self.progress)
        self.navi.addStretch(1)

        self.navi.addWidget(self.model_button)
        self.navi.addWidget(self.infer_button)
        self.navi.addWidget(self.save_button)


        # draw area
        draw_area = QHBoxLayout()
        draw_area.addWidget(self.main_canvas, 4)


        # Minimap area
        minimap_area = QVBoxLayout()
        minimap_area.setAlignment(Qt.AlignTop)
        mini_label = QLabel('Minimap')
        mini_label.setAlignment(Qt.AlignTop)

        minimap_area.addWidget(mini_label)

        minimap_ctrl = QHBoxLayout()
        ts_record = QHBoxLayout()

        minimap_ctrl.setAlignment(Qt.AlignTop)
        minimap_ctrl.addWidget(self.zoom_p_button)
        minimap_ctrl.addWidget(self.zoom_m_button)

        ts_record.setAlignment(Qt.AlignLeft)
        ts_record.addWidget(self.ts_label)
        ts_record.addWidget(self.ts_log)

        # Set the minimap area and console
        minimap_area.addLayout(minimap_ctrl)
        minimap_area.addWidget(self.minimap)
        minimap_area.addLayout(ts_record)
        minimap_area.addWidget(self.console)

        draw_area.addLayout(minimap_area, 2)


        # Set the main layout
        self.layout = QVBoxLayout()
        self.layout.addLayout(draw_area)
        self.layout.addWidget(self.tl_slider)
        self.layout.addLayout(self.navi)
        

        # timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)


        # self.retranslateUi(main_widget)
        # QtCore.QMetaObject.connectSlotsByName(main_widget)

    # def retranslateUi(self, main_widget):
    #     _translate = QtCore.QCoreApplication.translate
    #     main_widget.setWindowTitle(_translate("main_widget", "SR Label"))
    #     self.main_canvas.setText(_translate("main_widget", "TextLabel"))
    #     self.mini_label_2.setText(_translate("main_widget", "Minimap"))
    #     self.zoom_p_button_2.setText(_translate("main_widget", "Zoom +"))
    #     self.zoom_m_button_2.setText(_translate("main_widget", "Zoom -"))
    #     self.mini_map_2.setText(_translate("main_widget", "TextLabel"))
    #     self.ts_label_2.setText(_translate("main_widget", "Timestamp:"))
    #     self.play_button_2.setText(_translate("main_widget", "Play"))
    #     self.brush_size_label.setText(_translate("main_widget", "Brush Size: 1"))
    #     self.undo_button_2.setText(_translate("main_widget", "Undo"))
    #     self.reset_button_2.setText(_translate("main_widget", "Reset"))
    #     self.model_button.setText(_translate("main_widget", "Load Model"))
    #     self.infer_button.setText(_translate("main_widget", "Inference"))
    #     self.save_button.setText(_translate("main_widget", "Save"))
