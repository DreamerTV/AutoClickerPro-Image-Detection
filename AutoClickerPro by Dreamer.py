import sys
import time
import threading
import os
import ctypes
import json
import cv2
import numpy as np
from pynput import keyboard
import pyautogui
import webbrowser
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel, QWidget, QLineEdit, QMessageBox, QFormLayout, QGroupBox, QTextEdit, QFileDialog, QComboBox, QScrollBar, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

CHECK_INTERVAL = 5
IMAGE_DIR = r""
JSON_FILE = "config.json"

TARGET_POSITIONS = []
INPUT_TEXT = ""
DETECTION_IMAGES = []
COMMAND_INTERVAL = 1.0
CONFIDENCE = 0.7

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def validate_images():
    for path in DETECTION_IMAGES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"圖片文件缺失: {path}")
        if not path.lower().endswith('.png'):
            raise ValueError(f"非PNG格式文件: {path}")

def is_any_image_detected():
    try:
        screen = np.array(pyautogui.screenshot())
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        for image_path in DETECTION_IMAGES:
            template = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if template is None:
                raise FileNotFoundError(f"無法讀取圖片: {image_path}")
            result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= CONFIDENCE:
                return True
        return False
    except Exception as e:
        print(f"偵測異常: {str(e)}")
        return False

def perform_mouse_clicks():
    print("開始執行滑鼠點擊操作...")
    for item in TARGET_POSITIONS:
        if item[0] == "滑鼠":
            x, y, button, click_count, interval, window_title = item[1:]
            move_and_click(x, y, button, click_count, interval)
        elif item[0] == "鍵盤":
            text, interval, window_title = item[1:]
            pyautogui.write(text, interval=0.1)
    print("滑鼠點擊操作完成！")
    return True

def move_and_click(x, y, button, click_count, interval):
    pyautogui.moveTo(x, y)
    time.sleep(interval)
    if button != "無":
        for _ in range(click_count):
            if button == "左鍵":
                pyautogui.leftClick()
            else:
                pyautogui.rightClick()

def load_json_config():
    if not os.path.exists(JSON_FILE):
        save_json_config()
        return [], [], 1.0, 0.7, 5.0

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            return (
                config.get("target_positions", []),
                config.get("detection_images", []),
                config.get("command_interval", 1.0),
                config.get("confidence", 0.7),
                config.get("detection_interval", 5.0)
            )
    except (json.JSONDecodeError, FileNotFoundError):
        save_json_config()
        return [], [], 1.0, 0.7, 5.0

def save_json_config():
    config = {
        "target_positions": TARGET_POSITIONS,
        "detection_images": DETECTION_IMAGES,
        "command_interval": COMMAND_INTERVAL,
        "confidence": CONFIDENCE,
        "detection_interval": CHECK_INTERVAL
    }
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

class MouseClickApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能鍵鼠工具－圖像偵測 V1 by Dreamer")
        self.setGeometry(600, 40, 650, 1000)
        self.setWindowIcon(QIcon("icon.ico"))

        self.set_dark_mode()

        self.target_positions, self.detection_images, self.command_interval, self.confidence, self.detection_interval = load_json_config()
        global TARGET_POSITIONS, DETECTION_IMAGES, COMMAND_INTERVAL, CONFIDENCE, CHECK_INTERVAL
        TARGET_POSITIONS = self.target_positions.copy()
        DETECTION_IMAGES = self.detection_images.copy()
        COMMAND_INTERVAL = self.command_interval
        CONFIDENCE = self.confidence
        CHECK_INTERVAL = self.detection_interval

        self.input_text = INPUT_TEXT
        self.monitoring = False
        self.image_folder = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        self.label = QLabel("智能鍵鼠工具", self)
        self.label.setFont(QFont("Microsoft JhengHei", 24, QFont.Bold))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white;")
        self.layout.addWidget(self.label)

        self.status_label = QLabel("狀態：未啟用", self)
        self.status_label.setFont(QFont("Microsoft JhengHei", 16, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: red;")
        self.layout.addWidget(self.status_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area)

        self.target_group = QGroupBox("滑鼠座標設定", self)
        self.target_group.setStyleSheet("QGroupBox { color: white; }")
        self.target_layout = QFormLayout()

        self.xy_input = QLineEdit(self)
        self.xy_input.setPlaceholderText("輸入 X,Y 座標 (例如: 100,200)")
        self.xy_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.target_layout.addRow("X,Y 座標：", self.xy_input)

        self.mouse_button_combo = QComboBox(self)
        self.mouse_button_combo.addItem("無")
        self.mouse_button_combo.addItem("左鍵")
        self.mouse_button_combo.addItem("右鍵")
        self.mouse_button_combo.setStyleSheet("QComboBox { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 15px; border-left-width: 1px; border-left-color: #555; border-left-style: solid; border-top-right-radius: 5px; border-bottom-right-radius: 5px; } QComboBox:disabled { background-color: #555; }")
        self.mouse_button_combo.currentTextChanged.connect(self.update_mouse_button)
        self.target_layout.addRow("滑鼠按鍵：", self.mouse_button_combo)

        self.click_count_rate_input = QLineEdit(self)
        self.click_count_rate_input.setPlaceholderText("輸入點擊次數,點擊速率 (例如: 2,0.5)")
        self.click_count_rate_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.click_count_rate_input.setEnabled(False)
        self.target_layout.addRow("點擊次數,點擊速率：", self.click_count_rate_input)

        self.window_title_input = QLineEdit(self)
        self.window_title_input.setPlaceholderText("輸入視窗標題（選填）")
        self.window_title_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.target_layout.addRow("開啟視窗標題：", self.window_title_input)

        self.mouse_interval_input = QLineEdit(self)
        self.mouse_interval_input.setPlaceholderText("輸入指令間隔時間（秒）")
        self.mouse_interval_input.setText(str(self.command_interval))
        self.mouse_interval_input.textChanged.connect(self.update_command_interval)
        self.mouse_interval_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.target_layout.addRow("執行點擊時間（秒）：", self.mouse_interval_input)

        self.get_mouse_position_button = QPushButton("獲取滑鼠位置(X)", self)
        self.get_mouse_position_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 8px; font-family: 'Microsoft JhengHei'; font-size: 14px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")
        self.get_mouse_position_button.clicked.connect(self.get_mouse_position)
        self.target_layout.addRow(self.get_mouse_position_button)

        self.add_position_button = QPushButton("新增滑鼠指令", self)
        self.add_position_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 8px; font-family: 'Microsoft JhengHei'; font-size: 14px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")
        self.add_position_button.clicked.connect(self.add_target_position)
        self.target_layout.addRow(self.add_position_button)

        self.target_group.setLayout(self.target_layout)
        self.scroll_layout.addWidget(self.target_group)

        self.keyboard_group = QGroupBox("鍵盤輸入設定", self)
        self.keyboard_group.setStyleSheet("QGroupBox { color: white; }")
        self.keyboard_layout = QFormLayout()

        self.keyboard_input = QLineEdit(self)
        self.keyboard_input.setPlaceholderText("輸入數字/文字 (選填)")
        self.keyboard_input.setText(self.input_text)
        self.keyboard_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.keyboard_layout.addRow("鍵盤輸入：", self.keyboard_input)

        self.keyboard_combo_input = QLineEdit(self)
        self.keyboard_combo_input.setPlaceholderText("輸入鍵盤組合鍵或單個按鍵 (例如: ctrl+alt+del) (選填)")
        self.keyboard_combo_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.keyboard_layout.addRow("鍵盤組合鍵：", self.keyboard_combo_input)

        self.keyboard_window_title_input = QLineEdit(self)
        self.keyboard_window_title_input.setPlaceholderText("輸入視窗標題（選填）")
        self.keyboard_window_title_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.keyboard_layout.addRow("開啟視窗標題：", self.keyboard_window_title_input)

        self.keyboard_interval_input = QLineEdit(self)
        self.keyboard_interval_input.setPlaceholderText("輸入指令間隔時間（秒）")
        self.keyboard_interval_input.setText("1.0")
        self.keyboard_interval_input.textChanged.connect(self.update_command_interval)
        self.keyboard_interval_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.keyboard_layout.addRow("執行輸入時間（秒）：", self.keyboard_interval_input)

        self.add_keyboard_input_button = QPushButton("新增鍵盤指令", self)
        self.add_keyboard_input_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 8px; font-family: 'Microsoft JhengHei'; font-size: 14px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")
        self.add_keyboard_input_button.clicked.connect(self.add_keyboard_input)
        self.keyboard_layout.addRow(self.add_keyboard_input_button)

        self.keyboard_group.setLayout(self.keyboard_layout)
        self.scroll_layout.addWidget(self.keyboard_group)

        self.positions_group = QGroupBox("指令列表", self)
        self.positions_group.setStyleSheet("QGroupBox { color: white; }")
        self.positions_layout = QVBoxLayout()

        self.positions_listbox = QListWidget(self)
        self.positions_listbox.setStyleSheet("QListWidget { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QScrollBar:vertical { background: #555; width: 10px; margin: 0px 0px 0px 0px; } QScrollBar::handle:vertical { background: #888; min-height: 20px; border-radius: 5px; }")
        self.positions_listbox.setDragDropMode(QListWidget.InternalMove)
        self.positions_layout.addWidget(self.positions_listbox)

        self.delete_position_button = QPushButton("刪除選取的指令", self)
        self.delete_position_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 8px; font-family: 'Microsoft JhengHei'; font-size: 14px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")
        self.delete_position_button.clicked.connect(self.delete_target_position)
        self.positions_layout.addWidget(self.delete_position_button)

        self.positions_group.setLayout(self.positions_layout)
        self.scroll_layout.addWidget(self.positions_group)

        self.image_group = QGroupBox("偵測圖片列表", self)
        self.image_group.setStyleSheet("QGroupBox { color: white; }")
        self.image_layout = QVBoxLayout()

        self.image_listbox = QListWidget(self)
        self.image_listbox.setStyleSheet("QListWidget { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QScrollBar:vertical { background: #555; width: 10px; margin: 0px 0px 0px 0px; } QScrollBar::handle:vertical { background: #888; min-height: 20px; border-radius: 5px; }")
        self.image_listbox.setDragDropMode(QListWidget.NoDragDrop)
        self.image_layout.addWidget(self.image_listbox)

        self.add_image_button = QPushButton("新增偵測圖片", self)
        self.add_image_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 8px; font-family: 'Microsoft JhengHei'; font-size: 14px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")
        self.add_image_button.clicked.connect(self.add_detection_image)
        self.image_layout.addWidget(self.add_image_button)

        self.delete_image_button = QPushButton("刪除選取的圖片", self)
        self.delete_image_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 8px; font-family: 'Microsoft JhengHei'; font-size: 14px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")
        self.delete_image_button.clicked.connect(self.delete_detection_image)
        self.image_layout.addWidget(self.delete_image_button)

        self.confidence_combo = QComboBox(self)
        self.confidence_combo.addItem("高 (0.9)")
        self.confidence_combo.addItem("中 (0.7)")
        self.confidence_combo.addItem("低 (0.5)")
        self.confidence_combo.setCurrentText(f"中 ({self.confidence})")
        self.confidence_combo.currentTextChanged.connect(self.update_confidence)
        self.confidence_combo.setStyleSheet("QComboBox { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 15px; border-left-width: 1px; border-left-color: #555; border-left-style: solid; border-top-right-radius: 5px; border-bottom-right-radius: 5px; } QComboBox:disabled { background-color: #555; }")
        self.image_layout.addWidget(QLabel("偵測圖片精確度："))
        self.image_layout.addWidget(self.confidence_combo)

        self.detection_interval_input = QLineEdit(self)
        self.detection_interval_input.setPlaceholderText("輸入圖片偵測間隔時間（秒）")
        self.detection_interval_input.setText(str(self.detection_interval))
        self.detection_interval_input.textChanged.connect(self.update_detection_interval)
        self.detection_interval_input.setStyleSheet("QLineEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QLineEdit:disabled { background-color: #555; color: #888; }")
        self.image_layout.addWidget(QLabel("偵測圖片間隔時間（秒）："))
        self.image_layout.addWidget(self.detection_interval_input)

        self.image_group.setLayout(self.image_layout)
        self.scroll_layout.addWidget(self.image_group)

        self.monitor_button = QPushButton("開始偵測 (F2)", self)
        self.monitor_button.setFont(QFont("Microsoft JhengHei", 16, QFont.Bold))
        self.monitor_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 12px; font-family: 'Microsoft JhengHei'; font-size: 16px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")
        self.monitor_button.clicked.connect(self.toggle_monitoring)
        self.scroll_layout.addWidget(self.monitor_button)

        self.log_group = QGroupBox("執行紀錄日誌", self)
        self.log_group.setStyleSheet("QGroupBox { color: white; }")
        self.log_layout = QVBoxLayout()

        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { color: white; background-color: #333; border-radius: 5px; padding: 10px; border: 1px solid #555; font-size: 16px; } QScrollBar:vertical { background: #555; width: 10px; margin: 0px 0px 0px 0px; } QScrollBar::handle:vertical { background: #888; min-height: 20px; border-radius: 5px; }")
        self.log_layout.addWidget(self.log_text)

        self.author_button = QPushButton("作者的IG/YT", self)
        self.author_button.setFont(QFont("Microsoft JhengHei", 14, QFont.Bold))
        self.author_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 8px; font-family: 'Microsoft JhengHei'; font-size: 14px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")
        self.author_button.clicked.connect(lambda: webbrowser.open("https://links.dreamtvs.net/"))
        self.log_layout.addWidget(self.author_button)

        self.log_group.setLayout(self.log_layout)
        self.scroll_layout.addWidget(self.log_group)

        self.scroll_area.setStyleSheet("QScrollArea { background-color: #242424; } QScrollBar:vertical { background: #555; width: 10px; margin: 0px 0px 0px 0px; } QScrollBar::handle:vertical { background: #888; min-height: 20px; border-radius: 5px; }")

        self.central_widget.setLayout(self.layout)

        self.update_positions_listbox()
        self.update_image_listbox()

        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

    def set_dark_mode(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(0x24, 0x24, 0x24))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(0x24, 0x24, 0x24))
        dark_palette.setColor(QPalette.AlternateBase, QColor(0x24, 0x24, 0x24))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(0x24, 0x24, 0x24))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(dark_palette)

    def on_key_press(self, key):
        try:
            if key == keyboard.KeyCode.from_char('x'):
                self.get_mouse_position()
            elif key == keyboard.Key.f2:
                self.toggle_monitoring()
        except AttributeError:
            pass

    def save_config(self):
        global TARGET_POSITIONS, DETECTION_IMAGES, COMMAND_INTERVAL, CONFIDENCE, CHECK_INTERVAL
        TARGET_POSITIONS = self.target_positions
        DETECTION_IMAGES = self.detection_images
        COMMAND_INTERVAL = float(self.mouse_interval_input.text() or 1.0)
        CONFIDENCE = float(self.confidence_combo.currentText().split(" ")[-1].strip("()"))
        CHECK_INTERVAL = float(self.detection_interval_input.text())
        save_json_config()

    def update_command_interval(self):
        try:
            global COMMAND_INTERVAL
            COMMAND_INTERVAL = float(self.mouse_interval_input.text() or 1.0)
            self.save_config()
        except ValueError:
            pass

    def update_detection_interval(self):
        try:
            global CHECK_INTERVAL
            CHECK_INTERVAL = float(self.detection_interval_input.text())
            self.save_config()
        except ValueError:
            pass

    def update_mouse_button(self):
        if self.mouse_button_combo.currentText() == "無":
            self.click_count_rate_input.setEnabled(False)
        else:
            self.click_count_rate_input.setEnabled(True)

    def add_target_position(self):
        xy = self.xy_input.text()
        click_count_rate = self.click_count_rate_input.text()
        interval = self.mouse_interval_input.text()
        window_title = self.window_title_input.text()

        if xy and interval:
            try:
                x, y = map(int, xy.split(','))
                click_count, click_rate = map(float, click_count_rate.split(',')) if click_count_rate else (1, 0.0)
                click_count = int(click_count)
                interval = float(interval)
                button = self.mouse_button_combo.currentText()

                if window_title and not self.check_window_exists(window_title):
                    self.show_error_message(f"視窗 '{window_title}' 不存在！")
                    return

                self.target_positions.append(("滑鼠", x, y, button, click_count, interval, window_title))
                self.update_positions_listbox()
                self.xy_input.clear()
                self.click_count_rate_input.clear()
                self.window_title_input.clear()
                self.save_config()
            except ValueError:
                self.show_error_message("請輸入有效的數字！")
        else:
            self.show_error_message("請輸入 X,Y 座標和指令間隔時間！")

    def add_keyboard_input(self):
        text = self.keyboard_input.text()
        combo = self.keyboard_combo_input.text()
        interval = self.keyboard_interval_input.text()
        window_title = self.keyboard_window_title_input.text()

        if (text or combo) and interval:
            try:
                interval = float(interval)

                if window_title and not self.check_window_exists(window_title):
                    self.show_error_message(f"視窗 '{window_title}' 不存在！")
                    return

                if text:
                    self.target_positions.append(("鍵盤", text, interval, window_title))
                if combo:
                    self.target_positions.append(("鍵盤組合鍵", combo, interval, window_title))
                self.update_positions_listbox()
                self.keyboard_input.clear()
                self.keyboard_combo_input.clear()
                self.keyboard_window_title_input.clear()
                self.save_config()
            except ValueError:
                self.show_error_message("請輸入有效的數字！")
        else:
            self.show_error_message("請輸入鍵盤文字或組合鍵以及指令間隔時間！")

    def delete_target_position(self):
        selected_item = self.positions_listbox.currentRow()
        if selected_item == -1:
            self.show_error_message("請先選擇一個目標位置！")
            return

        self.target_positions.pop(selected_item)
        self.update_positions_listbox()
        self.save_config()

    def update_positions_listbox(self):
        self.positions_listbox.clear()
        for idx, item in enumerate(self.target_positions, start=1):
            if item[0] == "滑鼠":
                _, x, y, button, click_count, interval, window_title = item
                self.positions_listbox.addItem(f"位置 {idx}: ({x}, {y}), 按鍵: {button}, 點擊次數: {click_count}, 間隔: {interval}秒, 視窗: {window_title}")
            elif item[0] == "鍵盤":
                _, text, interval, window_title = item
                self.positions_listbox.addItem(f"位置 {idx}: 鍵盤輸入 {idx}: {text}, 間隔: {interval}秒, 視窗: {window_title}")
            elif item[0] == "鍵盤組合鍵":
                _, combo, interval, window_title = item
                self.positions_listbox.addItem(f"位置 {idx}: 鍵盤組合鍵 {idx}: {combo}, 間隔: {interval}秒, 視窗: {window_title}")

    def update_confidence(self):
        global CONFIDENCE
        CONFIDENCE = float(self.confidence_combo.currentText().split(" ")[-1].strip("()"))
        self.save_config()

    def toggle_monitoring(self):
        if self.monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        if len(self.target_positions) == 0:
            self.show_error_message("請先設定至少一個目標位置！")
            return

        if len(self.detection_images) == 0:
            self.show_error_message("請先新增至少一張偵測圖片！")
            return

        for item in self.target_positions:
            if item[0] == "滑鼠" and item[-1]:
                if not self.check_window_exists(item[-1]):
                    self.show_error_message(f"視窗 '{item[-1]}' 不存在！")
                    return
            elif item[0] == "鍵盤" and item[-1]:
                if not self.check_window_exists(item[-1]):
                    self.show_error_message(f"視窗 '{item[-1]}' 不存在！")
                    return
            elif item[0] == "鍵盤組合鍵" and item[-1]:
                if not self.check_window_exists(item[-1]):
                    self.show_error_message(f"視窗 '{item[-1]}' 不存在！")
                    return

        global TARGET_POSITIONS, INPUT_TEXT, COMMAND_INTERVAL
        TARGET_POSITIONS = self.target_positions
        INPUT_TEXT = self.keyboard_input.text()
        COMMAND_INTERVAL = float(self.mouse_interval_input.text() or 1.0)

        self.set_inputs_enabled(False)

        self.monitoring = True
        self.status_label.setText("狀態：偵測中")
        self.status_label.setStyleSheet("color: green;")
        self.monitor_button.setText("停止偵測 (F2)")
        self.monitor_button.setStyleSheet("QPushButton { background-color: #FF0000; color: white; border-radius: 5px; padding: 12px; font-family: 'Microsoft JhengHei'; font-size: 16px; border: 2px solid #CC0000; } QPushButton:hover { background-color: #CC0000; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")

        self.monitoring_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        self.status_label.setText("狀態：未啟用")
        self.status_label.setStyleSheet("color: red;")
        self.monitor_button.setText("開始偵測 (F2)")
        self.monitor_button.setStyleSheet("QPushButton { background-color: #0078D7; color: white; border-radius: 5px; padding: 12px; font-family: 'Microsoft JhengHei'; font-size: 16px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; } QPushButton:disabled { background-color: #555; border: 2px solid #555; }")

        self.set_inputs_enabled(True)

    def monitor(self):
        try:
            while self.monitoring:
                if is_any_image_detected():
                    self.log_text.append("偵測到圖片，開始執行指令...")
                    if perform_mouse_clicks():
                        self.log_text.append("指令執行完成，繼續偵測...")
                else:
                    self.log_text.append("偵測中...")
                time.sleep(float(self.detection_interval_input.text()))
                self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        except Exception as e:
            self.log_text.append(f"嚴重錯誤: {str(e)}")

    def check_window_exists(self, window_title):
        try:
            windows = pyautogui.getWindowsWithTitle(window_title)
            return len(windows) > 0
        except Exception:
            return False

    def get_mouse_position(self):
        x, y = pyautogui.position()
        self.xy_input.setText(f"{x},{y}")

    def set_inputs_enabled(self, enabled):
        self.xy_input.setEnabled(enabled)
        self.mouse_button_combo.setEnabled(enabled)
        self.click_count_rate_input.setEnabled(enabled)
        self.window_title_input.setEnabled(enabled)
        self.mouse_interval_input.setEnabled(enabled)
        self.get_mouse_position_button.setEnabled(enabled)
        self.add_position_button.setEnabled(enabled)
        self.keyboard_input.setEnabled(enabled)
        self.keyboard_combo_input.setEnabled(enabled)
        self.keyboard_interval_input.setEnabled(enabled)
        self.keyboard_window_title_input.setEnabled(enabled)
        self.add_keyboard_input_button.setEnabled(enabled)
        self.positions_listbox.setEnabled(enabled)
        self.delete_position_button.setEnabled(enabled)
        self.confidence_combo.setEnabled(enabled)
        self.detection_interval_input.setEnabled(enabled)

    def add_detection_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇圖片文件", IMAGE_DIR, "圖片文件 (*.png *.jpg *.bmp)")
        if file_path:
            image_name = os.path.basename(file_path).split('.')[0]
            if not image_name.isalnum():
                self.show_error_message("圖片名稱僅能使用英文和數字！")
                return

            if len(self.detection_images) == 0:
                self.image_folder = os.path.dirname(file_path)
            elif os.path.dirname(file_path) != self.image_folder:
                self.show_error_message("所有偵測圖片必須來自同一個資料夾！")
                return

            self.detection_images.append(file_path)
            self.update_image_listbox()
            self.save_config()

    def delete_detection_image(self):
        selected_item = self.image_listbox.currentRow()
        if selected_item == -1:
            self.show_error_message("請先選擇一個圖片！")
            return

        self.detection_images.pop(selected_item)
        self.update_image_listbox()
        self.save_config()

    def update_image_listbox(self):
        self.image_listbox.clear()
        for idx, image_path in enumerate(self.detection_images, start=1):
            self.image_listbox.addItem(f"圖片 {idx}: {os.path.basename(image_path)}")

    def show_error_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("錯誤")
        msg.setStyleSheet("QMessageBox { background-color: #242424; color: white; } QLabel { color: white; } QPushButton { color: white; background-color: #0078D7; border-radius: 5px; padding: 15px; font-family: '微軟正黑體'; font-size: 16px; border: 2px solid #005BB5; } QPushButton:hover { background-color: #005BB5; }")
        msg.exec_()

if __name__ == "__main__":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("com.mycompany.myapp")

    app = QApplication(sys.argv)
    icon_path = resource_path("icon.ico")
    app.setWindowIcon(QIcon(icon_path))

    window = MouseClickApp()
    window.show()

    sys.exit(app.exec_())