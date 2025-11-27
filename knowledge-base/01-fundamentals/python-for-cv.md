# Python for Computer Vision (Python Cho Thị Giác Máy Tính)

## 1. NumPy Cho Computer Vision

### A. Array Basics (Cơ Bản Về Mảng)

```python
import numpy as np

# Tạo mảng giống ảnh
image = np.zeros((480, 640, 3), dtype=np.uint8)
print(f"Shape: {image.shape}")  # (height, width, channels)

# Tô đầy bằng giá trị
white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

# Tạo gradient (dải màu)
gradient = np.linspace(0, 255, 256, dtype=np.uint8)
gradient_image = np.tile(gradient, (256, 1))
```

### B. Array Operations (Thao Tác Mảng)

```python
# Thao tác theo từng phần tử
image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Cộng (với clipping)
added = np.clip(image1.astype(np.int16) + image2.astype(np.int16), 0, 255).astype(np.uint8)

# Trừ
subtracted = np.clip(image1.astype(np.int16) - image2.astype(np.int16), 0, 255).astype(np.uint8)

# Nhân (scaling - điều chỉnh tỷ lệ)
scaled = np.clip(image1 * 1.5, 0, 255).astype(np.uint8)

# Blending (trộn ảnh)
alpha = 0.5
blended = np.clip(alpha * image1 + (1 - alpha) * image2, 0, 255).astype(np.uint8)
```

### C. Indexing and Slicing (Đánh Chỉ Số và Cắt Lát)

```python
# Truy cập pixel (y, x)
pixel = image[100, 200]  # Trả về [B, G, R]

# Truy cập kênh
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]

# ROI (Region of Interest - Vùng quan tâm)
roi = image[100:200, 150:250]  # [y1:y2, x1:x2]

# Đặt ROI thành giá trị
image[100:200, 150:250] = [255, 0, 0]  # Hình chữ nhật xanh dương

# Sao chép ROI
roi_copy = image[100:200, 150:250].copy()

# Boolean indexing (đánh chỉ số boolean)
bright_pixels = image[image > 200] = 255  # Đặt pixels sáng thành trắng
```

### D. Hàm Hữu Ích

```python
# Thống kê
mean = np.mean(image)
std = np.std(image)
min_val = np.min(image)
max_val = np.max(image)

# Thống kê theo từng kênh
channel_means = np.mean(image, axis=(0, 1))  # [B_mean, G_mean, R_mean]

# Histogram (biểu đồ)
hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

# Tìm chỉ số
bright_indices = np.where(image > 200)

# Giá trị duy nhất
unique_values = np.unique(image)

# Đếm giá trị khác không
non_zero_count = np.count_nonzero(image)
```

### E. Thao Tác Mảng

```python
# Reshape (thay đổi hình dạng)
flattened = image.reshape(-1, 3)  # (height*width, 3)

# Transpose axes (hoán vị trục)
transposed = np.transpose(image, (2, 0, 1))  # (C, H, W) cho PyTorch

# Flip (lật)
flipped_vertical = np.flip(image, axis=0)
flipped_horizontal = np.flip(image, axis=1)

# Xoay 90 độ
rotated = np.rot90(image)

# Stack images (ghép ảnh)
stacked_h = np.hstack([image1, image2])  # Theo chiều ngang
stacked_v = np.vstack([image1, image2])  # Theo chiều dọc

# Split channels (tách kênh)
b, g, r = np.split(image, 3, axis=2)

# Concatenate (nối)
combined = np.concatenate([image1, image2], axis=1)
```

---

## 2. Vectorization Techniques (Kỹ Thuật Vector Hóa)

### A. Tránh Vòng Lặp

```python
# ❌ Chậm: Sử dụng vòng lặp
def brighten_slow(image, value):
    result = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = min(255, image[i, j] + value)
    return result

# ✅ Nhanh: Vector hóa
def brighten_fast(image, value):
    return np.clip(image + value, 0, 255).astype(np.uint8)

# So sánh tốc độ
import time

image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

start = time.time()
result_slow = brighten_slow(image, 50)
print(f"Phiên bản vòng lặp: {time.time() - start:.3f}s")

start = time.time()
result_fast = brighten_fast(image, 50)
print(f"Vector hóa: {time.time() - start:.3f}s")

# Vector hóa nhanh hơn 100-1000 lần!
```

### B. Broadcasting (Phát Sóng)

```python
# Thêm giá trị khác nhau cho mỗi kênh
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Điều chỉnh theo từng kênh
adjustment = np.array([10, 20, 30])  # [B, G, R]

# Broadcasting tự động mở rộng adjustment thành (100, 100, 3)
adjusted = np.clip(image + adjustment, 0, 255).astype(np.uint8)

# Áp dụng mask
mask = image[:, :, 0] > 127  # Boolean mask từ kênh xanh dương

# Áp dụng mask lên tất cả kênh sử dụng broadcasting
image[mask] = [255, 0, 0]  # Đặt pixels đã mask thành xanh dương
```

### C. Lọc Hiệu Quả

```python
# Lọc theo khoảng màu (ví dụ: vật thể màu xanh dương)
lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 50, 50])

# So sánh vector hóa
mask = np.all((image >= lower_blue) & (image <= upper_blue), axis=2)

# Trích xuất vật thể màu xanh dương
blue_objects = image.copy()
blue_objects[~mask] = 0
```

---

## 3. Cấu Trúc Dữ Liệu Cho CV

### A. Collections (Bộ Sưu Tập)

```python
from collections import defaultdict, deque

# Theo dõi vật thể theo thời gian
object_tracking = defaultdict(dict)

# Thêm thông tin theo dõi
object_tracking[object_id] = {
    'first_seen': time.time(),
    'last_seen': time.time(),
    'positions': [(x, y)],
    'roi': 'Area 1'
}

# Buffer frames gần đây
frame_buffer = deque(maxlen=30)  # Giữ 30 frames cuối cùng

while True:
    ret, frame = cap.read()
    if ret:
        frame_buffer.append(frame)

    # Truy cập frames gần đây
    if len(frame_buffer) >= 3:
        current_frame = frame_buffer[-1]
        prev_frame = frame_buffer[-2]
        prev_prev_frame = frame_buffer[-3]
```

### B. Dataclasses Cho Cấu Hình

```python
from dataclasses import dataclass

@dataclass
class DetectionConfig:
    """Cấu hình cho phát hiện chuyển động"""
    method: str = "MOG2"
    threshold: int = 16
    history: int = 500
    detect_shadows: bool = True

    def to_dict(self):
        return {
            'method': self.method,
            'threshold': self.threshold,
            'history': self.history,
            'detect_shadows': self.detect_shadows
        }

# Sử dụng
config = DetectionConfig(method="KNN", threshold=400)
print(config.method)  # "KNN"
```

### C. Named Tuples (Tuple Có Tên)

```python
from collections import namedtuple

# Kết quả phát hiện
Detection = namedtuple('Detection', ['bbox', 'confidence', 'class_id'])

# Tạo detection
det = Detection(
    bbox=(100, 100, 50, 75),
    confidence=0.95,
    class_id=0
)

# Truy cập theo tên
x, y, w, h = det.bbox
print(f"Confidence: {det.confidence}")
```

---

## 4. File I/O (Đọc/Ghi File)

### A. JSON Cho Cấu Hình

```python
import json

# Lưu cấu hình
config = {
    'video': {
        'source': 'video.mp4',
        'skip_frames': 0
    },
    'motion': {
        'method': 'MOG2',
        'threshold': 16
    },
    'rois': [
        {
            'name': 'Entrance',
            'points': [[100, 100], [300, 100], [300, 300], [100, 300]]
        }
    ]
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Tải cấu hình
with open('config.json', 'r') as f:
    loaded_config = json.load(f)

print(loaded_config['motion']['method'])
```

### B. YAML (Được Khuyên Dùng)

```python
import yaml

# Lưu
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Tải
with open('config.yaml', 'r') as f:
    loaded_config = yaml.safe_load(f)
```

### C. Pickle Cho Objects

```python
import pickle

# Lưu đối tượng phức tạp
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Huấn luyện nó...
for frame in training_frames:
    bg_subtractor.apply(frame)

# Lưu mô hình đã huấn luyện (không phải tất cả đối tượng OpenCV đều hỗ trợ pickle!)
with open('bg_model.pkl', 'wb') as f:
    pickle.dump(bg_subtractor, f)

# Tải
with open('bg_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### D. CSV Cho Logs

```python
import csv
from datetime import datetime

# Ghi log phát hiện
with open('detections.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'ROI', 'X', 'Y', 'Width', 'Height'])

    # Ghi detections
    for detection in detections:
        writer.writerow([
            datetime.now().isoformat(),
            detection['roi_name'],
            detection['x'],
            detection['y'],
            detection['w'],
            detection['h']
        ])

# Đọc log
detections = []
with open('detections.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        detections.append(row)
```

---

## 5. Multiprocessing (Đa Xử Lý)

### A. Process Pool (Nhóm Tiến Trình)

```python
from multiprocessing import Pool
import cv2

def process_frame(args):
    """Xử lý frame đơn lẻ"""
    frame_path, output_path = args

    # Đọc
    frame = cv2.imread(frame_path)

    # Xử lý
    processed = your_processing_function(frame)

    # Lưu
    cv2.imwrite(output_path, processed)

    return output_path

# Chuẩn bị tham số
frame_paths = [f'frame_{i:04d}.jpg' for i in range(100)]
output_paths = [f'processed_{i:04d}.jpg' for i in range(100)]
args_list = list(zip(frame_paths, output_paths))

# Xử lý song song
with Pool(processes=4) as pool:
    results = pool.map(process_frame, args_list)

print(f"Đã xử lý {len(results)} frames")
```

### B. Threading Cho I/O

```python
import threading
import queue

# Frame queue (hàng đợi frame)
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue()

def capture_thread(source):
    """Chụp frames trong thread riêng"""
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Đặt frame vào queue
        frame_queue.put(frame)

    cap.release()
    frame_queue.put(None)  # Tín hiệu kết thúc

def processing_thread():
    """Xử lý frames từ queue"""
    while True:
        frame = frame_queue.get()

        if frame is None:
            break

        # Xử lý
        processed = your_processing_function(frame)

        # Đặt kết quả
        result_queue.put(processed)

    result_queue.put(None)

# Khởi động threads
capture = threading.Thread(target=capture_thread, args=('video.mp4',))
processing = threading.Thread(target=processing_thread)

capture.start()
processing.start()

# Tiêu thụ kết quả
while True:
    result = result_queue.get()
    if result is None:
        break

    # Hiển thị hoặc lưu kết quả
    cv2.imshow('Result', result)
    cv2.waitKey(1)

# Đợi threads
capture.join()
processing.join()
```

---

## 6. Xử Lý Lỗi

### A. Đọc Frame An Toàn

```python
def safe_read_frame(cap, max_retries=3):
    """Đọc frame an toàn với thử lại"""
    for attempt in range(max_retries):
        ret, frame = cap.read()

        if ret:
            return frame

        print(f"Đọc thất bại, lần thử {attempt + 1}/{max_retries}")
        time.sleep(0.1)

    raise RuntimeError("Không thể đọc frame sau khi thử lại")

# Sử dụng
try:
    frame = safe_read_frame(cap)
except RuntimeError as e:
    print(f"Lỗi: {e}")
    # Xử lý lỗi (khởi động lại capture, v.v.)
```

### B. Context Managers (Trình Quản Lý Ngữ Cảnh)

```python
class VideoCapture:
    """Context manager cho video capture"""

    def __init__(self, source):
        self.source = source
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở: {self.source}")
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# Sử dụng
try:
    with VideoCapture('video.mp4') as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Xử lý frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except RuntimeError as e:
    print(f"Lỗi: {e}")

# Dọn dẹp tự động!
```

### C. Validation (Kiểm Tra Tính Hợp Lệ)

```python
def validate_frame(frame):
    """Kiểm tra tính toàn vẹn của frame"""
    if frame is None:
        raise ValueError("Frame là None")

    if frame.size == 0:
        raise ValueError("Frame rỗng")

    if len(frame.shape) not in [2, 3]:
        raise ValueError(f"Shape frame không hợp lệ: {frame.shape}")

    if frame.dtype != np.uint8:
        raise ValueError(f"Dtype không hợp lệ: {frame.dtype}")

    return True

# Sử dụng
try:
    validate_frame(frame)
    # Xử lý frame
except ValueError as e:
    print(f"Frame không hợp lệ: {e}")
```

---

## 7. Logging (Ghi Log)

### A. Logging Cơ Bản

```python
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Sử dụng
logger.info("Bắt đầu xử lý")
logger.warning("Phát hiện FPS thấp: 12 FPS")
logger.error("Không thể đọc frame")

try:
    result = risky_operation()
except Exception as e:
    logger.exception("Có exception xảy ra")
```

### B. Custom Logger (Logger Tùy Chỉnh)

```python
class DetectionLogger:
    """Logger tùy chỉnh cho detections"""

    def __init__(self, log_file='detections.log'):
        self.logger = logging.getLogger('DetectionLogger')
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Format
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

    def log_detection(self, roi_name, bbox, confidence):
        """Ghi log sự kiện phát hiện"""
        self.logger.info(
            f"DETECTION | ROI: {roi_name} | "
            f"BBox: {bbox} | Confidence: {confidence:.2f}"
        )

    def log_intrusion(self, roi_name, duration):
        """Ghi log sự kiện xâm nhập"""
        self.logger.warning(
            f"INTRUSION | ROI: {roi_name} | Duration: {duration:.1f}s"
        )

# Sử dụng
det_logger = DetectionLogger()
det_logger.log_detection('Entrance', (100, 100, 50, 75), 0.95)
det_logger.log_intrusion('Entrance', 2.5)
```

---

## 8. Performance Profiling (Đo Hiệu Suất)

### A. Đo Thời Gian

```python
import time

def profile_function(func):
    """Decorator để đo thời gian thực thi hàm"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} mất {(end - start) * 1000:.2f}ms")
        return result
    return wrapper

@profile_function
def process_frame(frame):
    # Code xử lý
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# Sử dụng
result = process_frame(frame)
# Output: process_frame mất 12.34ms
```

### B. Code Profiler (Trình Đo Hiệu Suất Code)

```python
import cProfile
import pstats

def profile_code():
    """Đo hiệu suất toàn bộ pipeline xử lý"""
    profiler = cProfile.Profile()
    profiler.enable()

    # Code cần đo
    cap = cv2.VideoCapture('video.mp4')
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)

    profiler.disable()

    # In thống kê
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 10 hàm hàng đầu

# Chạy profiling
profile_code()
```

### C. Đo Bộ Nhớ

```python
import tracemalloc

# Bắt đầu theo dõi
tracemalloc.start()

# Code cần đo
frames = []
for i in range(100):
    frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    frames.append(frame)

# Lấy mức sử dụng bộ nhớ
current, peak = tracemalloc.get_traced_memory()
print(f"Bộ nhớ hiện tại: {current / 1024 / 1024:.2f} MB")
print(f"Bộ nhớ đỉnh: {peak / 1024 / 1024:.2f} MB")

# Dừng theo dõi
tracemalloc.stop()
```

---

## 9. Best Practices (Thực Hành Tốt)

### A. Type Hints (Gợi Ý Kiểu)

```python
from typing import Tuple, List, Optional
import numpy as np

def process_frame(
    frame: np.ndarray,
    threshold: int = 127
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Xử lý frame và phát hiện contours.

    Args:
        frame: Ảnh BGR đầu vào
        threshold: Giá trị ngưỡng nhị phân

    Returns:
        Tuple gồm (ảnh nhị phân, danh sách contours)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return binary, contours
```

### B. Documentation (Tài Liệu)

```python
def calculate_overlap(
    contour: np.ndarray,
    roi: np.ndarray,
    frame_shape: Tuple[int, int]
) -> float:
    """
    Tính chồng lấn giữa contour và ROI.

    Hàm này tạo binary masks cho cả contour và ROI,
    sau đó tính giao điểm trên diện tích contour.

    Args:
        contour: OpenCV contour (mảng Nx1x2)
        roi: Điểm polygon ROI (mảng Nx2)
        frame_shape: (height, width) của frame

    Returns:
        Tỷ lệ chồng lấn (0.0 đến 1.0)

    Example:
        >>> contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]]])
        >>> roi = np.array([[50, 50], [250, 50], [250, 250], [50, 250]])
        >>> overlap = calculate_overlap(contour, roi, (480, 640))
        >>> print(f"Chồng lấn: {overlap:.2%}")
        Chồng lấn: 75.00%
    """
    h, w = frame_shape

    # Tạo masks
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    # Tô đầy masks
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    cv2.fillPoly(roi_mask, [roi], 255)

    # Tính chồng lấn
    intersection = cv2.bitwise_and(contour_mask, roi_mask)
    intersection_area = cv2.countNonZero(intersection)
    contour_area = cv2.contourArea(contour)

    if contour_area > 0:
        return intersection_area / contour_area
    else:
        return 0.0
```

### C. Quản Lý Cấu Hình

```python
from pathlib import Path
import yaml

class Config:
    """Quản lý cấu hình tập trung"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = Path(config_path)
        self.config = self.load()

    def load(self) -> dict:
        """Tải cấu hình từ YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy config: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default=None):
        """Lấy giá trị cấu hình"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            value = value.get(k)
            if value is None:
                return default

        return value

    def save(self):
        """Lưu cấu hình"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# Sử dụng
config = Config('config.yaml')

video_source = config.get('video.source')
motion_method = config.get('motion.method', default='MOG2')
threshold = config.get('motion.threshold', default=16)
```

---

## 10. Testing (Kiểm Thử)

### A. Unit Tests (Kiểm Thử Đơn Vị)

```python
import unittest

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        """Tạo ảnh test"""
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_grayscale_conversion(self):
        """Kiểm thử chuyển đổi BGR sang grayscale"""
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)

        self.assertEqual(gray.shape, (100, 100))
        self.assertEqual(gray.dtype, np.uint8)

    def test_threshold(self):
        """Kiểm thử ngưỡng hóa"""
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Tất cả pixels phải là 0 hoặc 255
        unique_values = np.unique(binary)
        self.assertTrue(all(v in [0, 255] for v in unique_values))

    def tearDown(self):
        """Dọn dẹp"""
        del self.test_image

if __name__ == '__main__':
    unittest.main()
```

---

**Ngày tạo**: Tháng 1/2025
