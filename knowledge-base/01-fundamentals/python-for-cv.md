# Python for Computer Vision

## 1. NumPy cho Computer Vision

### A. Array Basics

```python
import numpy as np

# Create image-like array
image = np.zeros((480, 640, 3), dtype=np.uint8)
print(f"Shape: {image.shape}")  # (height, width, channels)

# Fill with value
white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

# Create gradient
gradient = np.linspace(0, 255, 256, dtype=np.uint8)
gradient_image = np.tile(gradient, (256, 1))
```

### B. Array Operations

```python
# Element-wise operations
image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Addition (với clipping)
added = np.clip(image1.astype(np.int16) + image2.astype(np.int16), 0, 255).astype(np.uint8)

# Subtraction
subtracted = np.clip(image1.astype(np.int16) - image2.astype(np.int16), 0, 255).astype(np.uint8)

# Multiplication (scaling)
scaled = np.clip(image1 * 1.5, 0, 255).astype(np.uint8)

# Blending
alpha = 0.5
blended = np.clip(alpha * image1 + (1 - alpha) * image2, 0, 255).astype(np.uint8)
```

### C. Indexing and Slicing

```python
# Access pixel (y, x)
pixel = image[100, 200]  # Returns [B, G, R]

# Access channel
blue_channel = image[:, :, 0]
green_channel = image[:, :, 1]
red_channel = image[:, :, 2]

# ROI (Region of Interest)
roi = image[100:200, 150:250]  # [y1:y2, x1:x2]

# Set ROI to value
image[100:200, 150:250] = [255, 0, 0]  # Blue rectangle

# Copy ROI
roi_copy = image[100:200, 150:250].copy()

# Boolean indexing
bright_pixels = image[image > 200] = 255  # Set bright pixels to white
```

### D. Useful Functions

```python
# Statistics
mean = np.mean(image)
std = np.std(image)
min_val = np.min(image)
max_val = np.max(image)

# Per-channel statistics
channel_means = np.mean(image, axis=(0, 1))  # [B_mean, G_mean, R_mean]

# Histogram
hist, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

# Find indices
bright_indices = np.where(image > 200)

# Unique values
unique_values = np.unique(image)

# Count non-zero
non_zero_count = np.count_nonzero(image)
```

### E. Array Manipulation

```python
# Reshape
flattened = image.reshape(-1, 3)  # (height*width, 3)

# Transpose axes
transposed = np.transpose(image, (2, 0, 1))  # (C, H, W) for PyTorch

# Flip
flipped_vertical = np.flip(image, axis=0)
flipped_horizontal = np.flip(image, axis=1)

# Rotate 90 degrees
rotated = np.rot90(image)

# Stack images
stacked_h = np.hstack([image1, image2])  # Horizontal
stacked_v = np.vstack([image1, image2])  # Vertical

# Split channels
b, g, r = np.split(image, 3, axis=2)

# Concatenate
combined = np.concatenate([image1, image2], axis=1)
```

---

## 2. Vectorization Techniques

### A. Avoid Loops

```python
# ❌ Slow: Using loops
def brighten_slow(image, value):
    result = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = min(255, image[i, j] + value)
    return result

# ✅ Fast: Vectorized
def brighten_fast(image, value):
    return np.clip(image + value, 0, 255).astype(np.uint8)

# Speed comparison
import time

image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

start = time.time()
result_slow = brighten_slow(image, 50)
print(f"Loop version: {time.time() - start:.3f}s")

start = time.time()
result_fast = brighten_fast(image, 50)
print(f"Vectorized: {time.time() - start:.3f}s")

# Vectorized is 100-1000x faster!
```

### B. Broadcasting

```python
# Add different values to each channel
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Adjustment per channel
adjustment = np.array([10, 20, 30])  # [B, G, R]

# Broadcasting automatically expands adjustment to (100, 100, 3)
adjusted = np.clip(image + adjustment, 0, 255).astype(np.uint8)

# Mask application
mask = image[:, :, 0] > 127  # Boolean mask from blue channel

# Apply mask to all channels using broadcasting
image[mask] = [255, 0, 0]  # Set masked pixels to blue
```

### C. Efficient Filtering

```python
# Color range filtering (e.g., blue objects)
lower_blue = np.array([100, 0, 0])
upper_blue = np.array([255, 50, 50])

# Vectorized comparison
mask = np.all((image >= lower_blue) & (image <= upper_blue), axis=2)

# Extract blue objects
blue_objects = image.copy()
blue_objects[~mask] = 0
```

---

## 3. Data Structures for CV

### A. Collections

```python
from collections import defaultdict, deque

# Track objects over time
object_tracking = defaultdict(dict)

# Add tracking info
object_tracking[object_id] = {
    'first_seen': time.time(),
    'last_seen': time.time(),
    'positions': [(x, y)],
    'roi': 'Area 1'
}

# Recent frames buffer
frame_buffer = deque(maxlen=30)  # Keep last 30 frames

while True:
    ret, frame = cap.read()
    if ret:
        frame_buffer.append(frame)

    # Access recent frames
    if len(frame_buffer) >= 3:
        current_frame = frame_buffer[-1]
        prev_frame = frame_buffer[-2]
        prev_prev_frame = frame_buffer[-3]
```

### B. Dataclasses for Configuration

```python
from dataclasses import dataclass

@dataclass
class DetectionConfig:
    """Configuration for motion detection"""
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

# Usage
config = DetectionConfig(method="KNN", threshold=400)
print(config.method)  # "KNN"
```

### C. Named Tuples

```python
from collections import namedtuple

# Detection result
Detection = namedtuple('Detection', ['bbox', 'confidence', 'class_id'])

# Create detection
det = Detection(
    bbox=(100, 100, 50, 75),
    confidence=0.95,
    class_id=0
)

# Access by name
x, y, w, h = det.bbox
print(f"Confidence: {det.confidence}")
```

---

## 4. File I/O

### A. JSON for Configuration

```python
import json

# Save configuration
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

# Load configuration
with open('config.json', 'r') as f:
    loaded_config = json.load(f)

print(loaded_config['motion']['method'])
```

### B. YAML (Recommended)

```python
import yaml

# Save
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# Load
with open('config.yaml', 'r') as f:
    loaded_config = yaml.safe_load(f)
```

### C. Pickle for Objects

```python
import pickle

# Save complex object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Train it...
for frame in training_frames:
    bg_subtractor.apply(frame)

# Save trained model (not all OpenCV objects support pickle!)
with open('bg_model.pkl', 'wb') as f:
    pickle.dump(bg_subtractor, f)

# Load
with open('bg_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

### D. CSV for Logs

```python
import csv
from datetime import datetime

# Write detection log
with open('detections.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'ROI', 'X', 'Y', 'Width', 'Height'])

    # Write detections
    for detection in detections:
        writer.writerow([
            datetime.now().isoformat(),
            detection['roi_name'],
            detection['x'],
            detection['y'],
            detection['w'],
            detection['h']
        ])

# Read log
detections = []
with open('detections.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        detections.append(row)
```

---

## 5. Multiprocessing

### A. Process Pool

```python
from multiprocessing import Pool
import cv2

def process_frame(args):
    """Process single frame"""
    frame_path, output_path = args

    # Read
    frame = cv2.imread(frame_path)

    # Process
    processed = your_processing_function(frame)

    # Save
    cv2.imwrite(output_path, processed)

    return output_path

# Prepare arguments
frame_paths = [f'frame_{i:04d}.jpg' for i in range(100)]
output_paths = [f'processed_{i:04d}.jpg' for i in range(100)]
args_list = list(zip(frame_paths, output_paths))

# Process in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_frame, args_list)

print(f"Processed {len(results)} frames")
```

### B. Threading for I/O

```python
import threading
import queue

# Frame queue
frame_queue = queue.Queue(maxsize=30)
result_queue = queue.Queue()

def capture_thread(source):
    """Capture frames in separate thread"""
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Put frame in queue
        frame_queue.put(frame)

    cap.release()
    frame_queue.put(None)  # Signal end

def processing_thread():
    """Process frames from queue"""
    while True:
        frame = frame_queue.get()

        if frame is None:
            break

        # Process
        processed = your_processing_function(frame)

        # Put result
        result_queue.put(processed)

    result_queue.put(None)

# Start threads
capture = threading.Thread(target=capture_thread, args=('video.mp4',))
processing = threading.Thread(target=processing_thread)

capture.start()
processing.start()

# Consume results
while True:
    result = result_queue.get()
    if result is None:
        break

    # Display or save result
    cv2.imshow('Result', result)
    cv2.waitKey(1)

# Wait for threads
capture.join()
processing.join()
```

---

## 6. Error Handling

### A. Robust Frame Reading

```python
def safe_read_frame(cap, max_retries=3):
    """Safely read frame with retries"""
    for attempt in range(max_retries):
        ret, frame = cap.read()

        if ret:
            return frame

        print(f"Read failed, attempt {attempt + 1}/{max_retries}")
        time.sleep(0.1)

    raise RuntimeError("Failed to read frame after retries")

# Usage
try:
    frame = safe_read_frame(cap)
except RuntimeError as e:
    print(f"Error: {e}")
    # Handle error (restart capture, etc.)
```

### B. Context Managers

```python
class VideoCapture:
    """Context manager for video capture"""

    def __init__(self, source):
        self.source = source
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.source}")
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

# Usage
try:
    with VideoCapture('video.mp4') as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except RuntimeError as e:
    print(f"Error: {e}")

# Cleanup automatic!
```

### C. Validation

```python
def validate_frame(frame):
    """Validate frame integrity"""
    if frame is None:
        raise ValueError("Frame is None")

    if frame.size == 0:
        raise ValueError("Frame is empty")

    if len(frame.shape) not in [2, 3]:
        raise ValueError(f"Invalid frame shape: {frame.shape}")

    if frame.dtype != np.uint8:
        raise ValueError(f"Invalid dtype: {frame.dtype}")

    return True

# Usage
try:
    validate_frame(frame)
    # Process frame
except ValueError as e:
    print(f"Invalid frame: {e}")
```

---

## 7. Logging

### A. Basic Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info("Processing started")
logger.warning("Low FPS detected: 12 FPS")
logger.error("Failed to read frame")

try:
    result = risky_operation()
except Exception as e:
    logger.exception("Exception occurred")
```

### B. Custom Logger

```python
class DetectionLogger:
    """Custom logger for detections"""

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
        """Log detection event"""
        self.logger.info(
            f"DETECTION | ROI: {roi_name} | "
            f"BBox: {bbox} | Confidence: {confidence:.2f}"
        )

    def log_intrusion(self, roi_name, duration):
        """Log intrusion event"""
        self.logger.warning(
            f"INTRUSION | ROI: {roi_name} | Duration: {duration:.1f}s"
        )

# Usage
det_logger = DetectionLogger()
det_logger.log_detection('Entrance', (100, 100, 50, 75), 0.95)
det_logger.log_intrusion('Entrance', 2.5)
```

---

## 8. Performance Profiling

### A. Time Measurement

```python
import time

def profile_function(func):
    """Decorator to profile function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start) * 1000:.2f}ms")
        return result
    return wrapper

@profile_function
def process_frame(frame):
    # Processing code
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# Usage
result = process_frame(frame)
# Output: process_frame took 12.34ms
```

### B. Code Profiler

```python
import cProfile
import pstats

def profile_code():
    """Profile entire processing pipeline"""
    profiler = cProfile.Profile()
    profiler.enable()

    # Code to profile
    cap = cv2.VideoCapture('video.mp4')
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

# Run profiling
profile_code()
```

### C. Memory Profiling

```python
import tracemalloc

# Start tracing
tracemalloc.start()

# Code to profile
frames = []
for i in range(100):
    frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    frames.append(frame)

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

# Stop tracing
tracemalloc.stop()
```

---

## 9. Best Practices

### A. Type Hints

```python
from typing import Tuple, List, Optional
import numpy as np

def process_frame(
    frame: np.ndarray,
    threshold: int = 127
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Process frame and detect contours.

    Args:
        frame: Input BGR image
        threshold: Binary threshold value

    Returns:
        Tuple of (binary image, list of contours)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return binary, contours
```

### B. Documentation

```python
def calculate_overlap(
    contour: np.ndarray,
    roi: np.ndarray,
    frame_shape: Tuple[int, int]
) -> float:
    """
    Calculate overlap between contour and ROI.

    This function creates binary masks for both the contour and ROI,
    then calculates the intersection over contour area.

    Args:
        contour: OpenCV contour (Nx1x2 array)
        roi: ROI polygon points (Nx2 array)
        frame_shape: (height, width) of frame

    Returns:
        Overlap ratio (0.0 to 1.0)

    Example:
        >>> contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]]])
        >>> roi = np.array([[50, 50], [250, 50], [250, 250], [50, 250]])
        >>> overlap = calculate_overlap(contour, roi, (480, 640))
        >>> print(f"Overlap: {overlap:.2%}")
        Overlap: 75.00%
    """
    h, w = frame_shape

    # Create masks
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    # Fill masks
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    cv2.fillPoly(roi_mask, [roi], 255)

    # Calculate overlap
    intersection = cv2.bitwise_and(contour_mask, roi_mask)
    intersection_area = cv2.countNonZero(intersection)
    contour_area = cv2.contourArea(contour)

    if contour_area > 0:
        return intersection_area / contour_area
    else:
        return 0.0
```

### C. Configuration Management

```python
from pathlib import Path
import yaml

class Config:
    """Centralized configuration management"""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = Path(config_path)
        self.config = self.load()

    def load(self) -> dict:
        """Load configuration from YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            value = value.get(k)
            if value is None:
                return default

        return value

    def save(self):
        """Save configuration"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# Usage
config = Config('config.yaml')

video_source = config.get('video.source')
motion_method = config.get('motion.method', default='MOG2')
threshold = config.get('motion.threshold', default=16)
```

---

## 10. Testing

### A. Unit Tests

```python
import unittest

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        """Create test image"""
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_grayscale_conversion(self):
        """Test BGR to grayscale conversion"""
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)

        self.assertEqual(gray.shape, (100, 100))
        self.assertEqual(gray.dtype, np.uint8)

    def test_threshold(self):
        """Test thresholding"""
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # All pixels should be 0 or 255
        unique_values = np.unique(binary)
        self.assertTrue(all(v in [0, 255] for v in unique_values))

    def tearDown(self):
        """Cleanup"""
        del self.test_image

if __name__ == '__main__':
    unittest.main()
```

---

**Ngày tạo**: Tháng 1/2025
