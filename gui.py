import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QSlider, QFileDialog, QListWidget, QListWidgetItem, QLineEdit, QHBoxLayout, QWidget, QToolBar, QAction, QSplitter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_paths = []
        self.filtered_image_paths = []
        self.current_image_index = -1
        self.original_image = None
        self.processed_image = None
        self.detected_objects = []

        self.scale_percent = 20
        self.padding = 25

        self.canny_low = 50
        self.canny_high = 150

        self.kernel_size = 5
        self.min_contour_area = 1000

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Objekterkennung')
        self.setGeometry(100, 100, 1200, 800)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.toolbar = QToolBar("Toolbar")
        self.addToolBar(self.toolbar)

        open_folder_action = QAction("Ordner öffnen", self)
        open_folder_action.triggered.connect(self.open_folder)
        self.toolbar.addAction(open_folder_action)

        prev_action = QAction("Zurück", self)
        prev_action.triggered.connect(self.prev_image)
        self.toolbar.addAction(prev_action)

        next_action = QAction("Vor", self)
        next_action.triggered.connect(self.next_image)
        self.toolbar.addAction(next_action)

        self.file_list = QListWidget(self)
        self.file_list.currentRowChanged.connect(self.on_file_selected)

        self.search_box = QLineEdit(self)
        self.search_box.setPlaceholderText("Datei suchen...")
        self.search_box.textChanged.connect(self.filter_files)

        self.filename_label = QLabel(self)
        self.filename_label.setAlignment(Qt.AlignCenter)

        self.canny_low_slider = QSlider(Qt.Horizontal, self)
        self.canny_low_slider.setRange(0, 255)
        self.canny_low_slider.setValue(self.canny_low)
        self.canny_low_slider.setTickPosition(QSlider.TicksBelow)
        self.canny_low_slider.setTickInterval(5)
        self.canny_low_slider.valueChanged.connect(self.slider_changed)
        self.canny_low_label = QLabel(f"Canny Low: {self.canny_low}", self)

        self.canny_high_slider = QSlider(Qt.Horizontal, self)
        self.canny_high_slider.setRange(0, 255)
        self.canny_high_slider.setValue(self.canny_high)
        self.canny_high_slider.setTickPosition(QSlider.TicksBelow)
        self.canny_high_slider.setTickInterval(5)
        self.canny_high_slider.valueChanged.connect(self.slider_changed)
        self.canny_high_label = QLabel(f"Canny High: {self.canny_high}", self)

        self.kernel_size_slider = QSlider(Qt.Horizontal, self)
        self.kernel_size_slider.setRange(1, 50)
        self.kernel_size_slider.setValue(self.kernel_size)
        self.kernel_size_slider.setTickPosition(QSlider.TicksBelow)
        self.kernel_size_slider.setTickInterval(1)
        self.kernel_size_slider.valueChanged.connect(self.slider_changed)
        self.kernel_size_label = QLabel(f"Kernel Size: {self.kernel_size}", self)

        self.min_contour_area_slider = QSlider(Qt.Horizontal, self)
        self.min_contour_area_slider.setRange(100, 10000)
        self.min_contour_area_slider.setValue(self.min_contour_area)
        self.min_contour_area_slider.setTickPosition(QSlider.TicksBelow)
        self.min_contour_area_slider.setTickInterval(100)
        self.min_contour_area_slider.valueChanged.connect(self.slider_changed)
        self.min_contour_area_label = QLabel(f"Min Contour Area: {self.min_contour_area}", self)

        self.padding_slider = QSlider(Qt.Horizontal, self)
        self.padding_slider.setRange(0, 100)
        self.padding_slider.setValue(self.padding)
        self.padding_slider.setTickPosition(QSlider.TicksBelow)
        self.padding_slider.setTickInterval(5)
        self.padding_slider.valueChanged.connect(self.slider_changed)
        self.padding_label = QLabel(f"Padding: {self.padding}", self)

        self.object_listbox = QListWidget(self)
        self.object_listbox.setSelectionMode(QListWidget.ExtendedSelection)

        self.save_button = QPushButton('Auswahl speichern', self)
        self.save_button.clicked.connect(self.save_selected_objects)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.search_box)
        left_layout.addWidget(self.file_list)
        left_layout.addWidget(self.filename_label)
        left_layout.addWidget(self.canny_low_label)
        left_layout.addWidget(self.canny_low_slider)
        left_layout.addWidget(self.canny_high_label)
        left_layout.addWidget(self.canny_high_slider)
        left_layout.addWidget(self.kernel_size_label)
        left_layout.addWidget(self.kernel_size_slider)
        left_layout.addWidget(self.min_contour_area_label)
        left_layout.addWidget(self.min_contour_area_slider)
        left_layout.addWidget(self.padding_label)
        left_layout.addWidget(self.padding_slider)
        left_layout.addWidget(self.object_listbox)
        left_layout.addWidget(self.save_button)

        left_container = QWidget()
        left_container.setLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)

        right_container = QWidget()
        right_container.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([300, 900])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(splitter)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Ordner auswählen')
        if folder_path:
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            self.image_paths.sort()
            self.filter_files("")

    def filter_files(self, text):
        self.file_list.clear()
        self.filtered_image_paths = [path for path in self.image_paths if text.lower() in os.path.basename(path).lower()]
        for image_path in self.filtered_image_paths:
            self.file_list.addItem(os.path.basename(image_path))
        if self.filtered_image_paths:
            self.file_list.setCurrentRow(0)

    def on_file_selected(self, index):
        if index != -1:
            self.current_image_index = index
            self.load_image()

    def load_image(self):
        if 0 <= self.current_image_index < len(self.filtered_image_paths):
            self.image_path = self.filtered_image_paths[self.current_image_index]
            self.filename_label.setText(os.path.basename(self.image_path))
            self.original_image = cv2.imread(self.image_path)
            self.show_image(self.original_image)
            self.update_image()

    def show_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def update_image(self):
        if self.original_image is None:
            return

        self.canny_low = self.canny_low_slider.value()
        self.canny_high = self.canny_high_slider.value()
        self.kernel_size = self.kernel_size_slider.value()
        self.min_contour_area = self.min_contour_area_slider.value()
        self.padding = self.padding_slider.value()

        self.canny_low_label.setText(f"Canny Low: {self.canny_low}")
        self.canny_high_label.setText(f"Canny High: {self.canny_high}")
        self.kernel_size_label.setText(f"Kernel Size: {self.kernel_size}")
        self.min_contour_area_label.setText(f"Min Contour Area: {self.min_contour_area}")
        self.padding_label.setText(f"Padding: {self.padding}")

        resized_image = self.resize_image(self.original_image)
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.detected_objects = [contour for contour in contours if cv2.contourArea(contour) > self.min_contour_area]

        self.processed_image = resized_image.copy()
        self.object_listbox.clear()

        for i, contour in enumerate(self.detected_objects):
            x, y, w, h = cv2.boundingRect(contour)
            x_pad = max(0, x - self.padding)
            y_pad = max(0, y - self.padding)
            w_pad = min(self.processed_image.shape[1] - x_pad, w + 2 * self.padding)
            h_pad = min(self.processed_image.shape[0] - y_pad, h + 2 * self.padding)
            cv2.rectangle(self.processed_image, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (0, 255, 0), 2)
            cv2.putText(self.processed_image, f"{i + 1}", (x_pad, y_pad - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.object_listbox.addItem(f"Objekt {i + 1}")

        self.show_image(self.processed_image)

    def resize_image(self, image):
        height, width = image.shape[:2]
        aspect_ratio = width / height
        max_height = self.height() - 200  # Adjust to fit the window
        max_width = self.width() - 400  # Adjust to fit the window
        if width > max_width or height > max_height:
            if aspect_ratio > 1:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            dim = (new_width, new_height)
            return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return image

    def save_selected_objects(self):
        if self.original_image is None:
            return

        selected_indices = self.object_listbox.selectedIndexes()
        for index in selected_indices:
            i = index.row()
            contour = self.detected_objects[i]
            x, y, w, h = cv2.boundingRect(contour)

            x_orig = int(x / self.scale_percent * 100) - self.padding
            y_orig = int(y / self.scale_percent * 100) - self.padding
            w_orig = int(w / self.scale_percent * 100) + 2 * self.padding
            h_orig = int(h / self.scale_percent * 100) + 2 * self.padding

            x_orig = max(0, x_orig)
            y_orig = max(0, y_orig)
            w_orig = min(self.original_image.shape[1] - x_orig, w_orig)
            h_orig = min(self.original_image.shape[0] - y_orig, h_orig)

            cropped = self.original_image[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]
            output_path = os.path.join('out', f'{os.path.splitext(os.path.basename(self.image_path))[0]}_{i + 1}.png')
            cv2.imwrite(output_path, cropped)

        print("Ausgewählte Objekte wurden gespeichert.")

    def prev_image(self):
        if self.filtered_image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def next_image(self):
        if self.filtered_image_paths and self.current_image_index < len(self.filtered_image_paths) - 1:
            self.current_image_index += 1
            self.load_image()

    def slider_changed(self):
        self.update_image()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ObjectDetectionApp()
    ex.show()
    sys.exit(app.exec_())
