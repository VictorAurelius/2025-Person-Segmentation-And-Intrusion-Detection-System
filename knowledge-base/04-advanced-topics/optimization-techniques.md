# Optimization Techniques (Kỹ Thuật Tối Ưu Hóa)

## 1. Profiling và Benchmarking

### A. Đo Thời Gian Code

```python
import time
import cv2

def benchmark_function(func, *args, iterations=100):
    """Benchmark thời gian thực thi hàm"""
    times = []

    for _ in range(iterations):
        start = time.time()
        result = func(*args)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"Hàm: {func.__name__}")
    print(f"  Trung bình: {avg_time * 1000:.2f}ms")
    print(f"  Tối thiểu: {min_time * 1000:.2f}ms")
    print(f"  Tối đa: {max_time * 1000:.2f}ms")

    return result


# Sử dụng
image = cv2.imread('image.jpg')

def method1(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def method2(img):
    return cv2.blur(img, (5, 5))

benchmark_function(method1, image)
benchmark_function(method2, image)
```

### B. Sử Dụng timeit

```python
import timeit

# Setup code
setup = """
import cv2
import numpy as np
image = cv2.imread('image.jpg')
"""

# Phương pháp 1
code1 = """
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
"""

# Phương pháp 2
code2 = """
gray = np.mean(image, axis=2).astype(np.uint8)
"""

time1 = timeit.timeit(code1, setup=setup, number=1000)
time2 = timeit.timeit(code2, setup=setup, number=1000)

print(f"cv2.cvtColor: {time1:.4f}s")
print(f"np.mean: {time2:.4f}s")
print(f"Nhanh hơn: {'cv2.cvtColor' if time1 < time2 else 'np.mean'} bằng {abs(time1 - time2) / min(time1, time2) * 100:.1f}%")
```

### C. Line Profiler

```python
# Cài đặt: pip install line_profiler

from line_profiler import LineProfiler
import cv2

def process_image(image):
    """Hàm cần profile"""
    # Dòng 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Dòng 2
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dòng 3
    edges = cv2.Canny(blurred, 50, 150)

    # Dòng 4
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# Profile
profiler = LineProfiler()
profiler.add_function(process_image)

image = cv2.imread('image.jpg')

# Chạy với profiler
profiler.enable()
result = process_image(image)
profiler.disable()

# In thống kê
profiler.print_stats()
```

---

## 2. Thay Đổi Kích Thước Ảnh

### A. Giảm Độ Phân Giải

```python
# Gốc: 1920×1080
image = cv2.imread('image.jpg')
print(f"Gốc: {image.shape}")

# Resize về 50%
scale = 0.5
small = cv2.resize(image, None, fx=scale, fy=scale)
print(f"Đã resize: {small.shape}")

# Xử lý ảnh nhỏ (nhanh hơn 4 lần!)
processed = process_frame(small)

# Resize lại nếu cần
large = cv2.resize(processed, (image.shape[1], image.shape[0]))
```

**So Sánh Tốc Độ:**
```
Độ Phân Giải   Pixels     Tốc Độ Tương Đối
1920×1080      2,073,600   1x (cơ sở)
1280×720       921,600     Nhanh hơn 2.25x
960×540        518,400     Nhanh hơn 4x
640×480        307,200     Nhanh hơn 6.75x
```

### B. Resize Thông Minh

```python
def smart_resize(image, target_pixels=500000):
    """Resize về số pixels mục tiêu"""
    h, w = image.shape[:2]
    current_pixels = h * w

    if current_pixels <= target_pixels:
        return image

    # Tính toán scale
    scale = (target_pixels / current_pixels) ** 0.5

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    return resized


# Tự động resize ảnh lớn
image = cv2.imread('large_image.jpg')
optimized = smart_resize(image, target_pixels=500000)

print(f"Gốc: {image.shape[1]}×{image.shape[0]}")
print(f"Đã tối ưu: {optimized.shape[1]}×{optimized.shape[0]}")
```

---

## 3. Xử Lý ROI

### A. Chỉ Xử Lý ROI

```python
# Xử lý toàn bộ frame (chậm)
def process_full_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


# Xử lý chỉ ROI (nhanh)
def process_roi_only(frame, roi):
    """Chỉ xử lý vùng quan tâm"""
    x, y, w, h = roi

    # Trích xuất ROI
    roi_frame = frame[y:y+h, x:x+w]

    # Xử lý ROI
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Tạo kết quả kích thước đầy đủ
    result = np.zeros(frame.shape[:2], dtype=np.uint8)
    result[y:y+h, x:x+w] = edges

    return result


# Benchmark
roi = (400, 200, 800, 600)  # Chỉ xử lý vùng 800×600 của 1920×1080

time1 = timeit.timeit(lambda: process_full_frame(frame), number=100)
time2 = timeit.timeit(lambda: process_roi_only(frame, roi), number=100)

print(f"Toàn frame: {time1:.4f}s")
print(f"Chỉ ROI: {time2:.4f}s")
print(f"Tăng tốc: {time1 / time2:.2f}x")
```

---

## 4. Frame Skipping (Bỏ Qua Frame)

### A. Xử Lý Mỗi N Frames

```python
class FrameSkipProcessor:
    """Xử lý mỗi N frames, tái sử dụng kết quả"""

    def __init__(self, skip_frames=2):
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.last_result = None

    def process(self, frame):
        """Xử lý frame hoặc trả về kết quả cache"""
        self.frame_count += 1

        # Xử lý frame này?
        if self.frame_count % self.skip_frames == 0:
            self.last_result = expensive_processing(frame)

        return self.last_result


# Sử dụng
processor = FrameSkipProcessor(skip_frames=3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý mỗi 3 frames
    result = processor.process(frame)

    cv2.imshow('Result', result)
```

**Cải Thiện FPS:**
```
skip_frames = 1:  15 FPS (xử lý tất cả)
skip_frames = 2:  25 FPS (nhanh hơn 67%)
skip_frames = 3:  35 FPS (nhanh hơn 133%)
```

---

## 5. Caching (Bộ Nhớ Đệm)

### A. LRU Cache

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def expensive_computation(image_hash):
    """Tính toán tốn kém với caching"""
    # Mô phỏng thao tác tốn kém
    time.sleep(0.1)
    return image_hash


def process_with_cache(image):
    """Xử lý ảnh với caching"""
    # Tạo hash
    image_hash = hashlib.md5(image.tobytes()).hexdigest()

    # Sử dụng kết quả cache nếu có
    result = expensive_computation(image_hash)

    return result


# Lần gọi đầu: chậm (cache miss)
start = time.time()
result1 = process_with_cache(image)
print(f"Lần gọi đầu: {time.time() - start:.4f}s")

# Lần gọi thứ hai: nhanh (cache hit)
start = time.time()
result2 = process_with_cache(image)
print(f"Lần gọi thứ hai: {time.time() - start:.4f}s")
```

### B. Result Caching

```python
class ResultCache:
    """Cache kết quả xử lý"""

    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size

    def get(self, key):
        """Lấy kết quả cache"""
        return self.cache.get(key)

    def put(self, key, value):
        """Lưu kết quả"""
        # LRU đơn giản: xóa cũ nhất nếu đầy
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        self.cache[key] = value

    def clear(self):
        """Xóa cache"""
        self.cache.clear()


# Sử dụng
cache = ResultCache(max_size=100)

def process_frame_cached(frame, frame_number):
    """Xử lý với caching"""

    # Kiểm tra cache
    cached = cache.get(frame_number)
    if cached is not None:
        return cached

    # Xử lý
    result = expensive_processing(frame)

    # Lưu trữ
    cache.put(frame_number, result)

    return result
```

---

## 6. Tối Ưu Hóa NumPy

### A. Vectorization (Véc-tơ Hóa)

```python
import numpy as np

# Chậm: Vòng lặp
def brighten_loop(image, value):
    result = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = min(255, image[i, j] + value)
    return result


# Nhanh: Vectorized
def brighten_vectorized(image, value):
    return np.clip(image + value, 0, 255).astype(np.uint8)


# Benchmark
image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

time1 = timeit.timeit(lambda: brighten_loop(image, 50), number=10)
time2 = timeit.timeit(lambda: brighten_vectorized(image, 50), number=10)

print(f"Vòng lặp: {time1:.4f}s")
print(f"Vectorized: {time2:.4f}s")
print(f"Tăng tốc: {time1 / time2:.0f}x")  # ~1000x nhanh hơn!
```

### B. In-Place Operations (Thao Tác Tại Chỗ)

```python
# Tạo bản sao (chậm hơn, nhiều bộ nhớ hơn)
def process_copy(image):
    result = image.copy()
    result = result * 1.5
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


# In-place (nhanh hơn, ít bộ nhớ hơn)
def process_inplace(image):
    image = image.astype(np.float32)
    image *= 1.5
    np.clip(image, 0, 255, out=image)
    return image.astype(np.uint8)
```

### C. Kiểu Dữ Liệu

```python
# Sử dụng dtype phù hợp
image_uint8 = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)   # 6.2 MB
image_float32 = image_uint8.astype(np.float32)  # 24.8 MB (lớn gấp 4 lần!)
image_float64 = image_uint8.astype(np.float64)  # 49.6 MB (lớn gấp 8 lần!)

# Chuyển đổi chỉ khi cần thiết
def process_smart(image):
    # Xử lý trong uint8 khi có thể
    result = cv2.GaussianBlur(image, (5, 5), 0)

    # Chuyển sang float chỉ cho các thao tác cần nó
    if need_float_precision:
        result = result.astype(np.float32)
        result *= 1.5
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result
```

---

## 7. Tối Ưu Hóa OpenCV

### A. Sử Dụng Hàm Phù Hợp

```python
# Chậm: Nhiều thao tác
def convert_slow(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    return binary


# Nhanh: Thao tác kết hợp
def convert_fast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary
```

### B. Tránh Chuyển Đổi Lặp Lại

```python
# Không hiệu quả
for _ in range(100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển đổi mỗi lần
    process(gray)


# Hiệu quả
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển đổi một lần
for _ in range(100):
    process(gray)
```

### C. Sử Dụng Hàm Tích Hợp

```python
# Chậm: Triển khai thủ công
def manual_blur(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    result = cv2.filter2D(image, -1, kernel)
    return result


# Nhanh: Hàm tích hợp (đã tối ưu)
def builtin_blur(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))


# Tích hợp nhanh hơn 5-10 lần!
```

---

## 8. Quản Lý Bộ Nhớ

### A. Cấp Phát Trước Arrays

```python
# Chậm: Tạo array mới mỗi lần
def process_slow(num_frames):
    results = []
    for i in range(num_frames):
        result = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results.append(result)
    return results


# Nhanh: Cấp phát trước
def process_fast(num_frames):
    results = np.zeros((num_frames, 1080, 1920, 3), dtype=np.uint8)
    for i in range(num_frames):
        results[i] = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return results
```

### B. Giải Phóng Bộ Nhớ

```python
import gc

def process_large_batch(images):
    """Xử lý batch lớn với quản lý bộ nhớ"""

    results = []

    for i, image in enumerate(images):
        # Xử lý
        result = expensive_processing(image)
        results.append(result)

        # Định kỳ giải phóng bộ nhớ
        if i % 100 == 0:
            gc.collect()

    return results
```

---

## 9. GPU Acceleration (Tăng Tốc GPU)

### A. Kiểm Tra CUDA Có Sẵn

```python
# Kiểm tra OpenCV có build với CUDA không
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"CUDA có sẵn: {cuda_available}")

if cuda_available:
    print(f"Số thiết bị CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}")
```

### B. Xử Lý GPU

```python
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Upload lên GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # Xử lý trên GPU
    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
    gpu_blurred = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0
    ).apply(gpu_gray)

    # Download kết quả
    result = gpu_blurred.download()

else:
    # Fallback sang CPU
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(gray, (5, 5), 0)
```

---

## 10. Lựa Chọn Thuật Toán

### A. Chọn Thuật Toán Đúng

```python
# Kịch bản: Cần làm mờ ảnh

# Kernel nhỏ (3×3): cv2.blur (nhanh nhất)
if kernel_size <= 3:
    blurred = cv2.blur(image, (3, 3))

# Kernel trung bình (5×5 đến 11×11): cv2.GaussianBlur (cân bằng)
elif kernel_size <= 11:
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Kernel lớn (>11): cv2.boxFilter + nhiều lần (nhanh hơn)
else:
    blurred = image
    for _ in range(3):
        blurred = cv2.boxFilter(blurred, -1, (5, 5))
```

### B. Đánh Đổi

```python
# Kịch bản: Phát hiện chuyển động

# Độ chính xác cao nhất (chậm nhất)
def method_accurate(frame, bg_model):
    return cv2.createBackgroundSubtractorKNN().apply(frame)


# Cân bằng (khuyến nghị)
def method_balanced(frame, bg_model):
    return cv2.createBackgroundSubtractorMOG2().apply(frame)


# Nhanh nhất (độ chính xác thấp hơn)
def method_fast(frame, prev_frame):
    diff = cv2.absdiff(frame, prev_frame)
    _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return binary
```

---

## 11. Ví Dụ Tối Ưu Hóa Hoàn Chỉnh

```python
class OptimizedVideoProcessor:
    """Bộ xử lý video được tối ưu cao"""

    def __init__(self, video_path, target_fps=30):
        self.video_path = video_path
        self.target_fps = target_fps

        # Tối ưu hóa
        self.scale = 0.5  # Resize về 50%
        self.skip_frames = 2  # Xử lý mỗi frame thứ 2
        self.roi = None  # Chỉ xử lý ROI cụ thể

        # Caching
        self.cache = {}
        self.last_result = None

    def process(self):
        """Xử lý video với tối ưu hóa"""

        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        start_time = time.time()

        # Lấy thuộc tính
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] Gốc: {width}×{height} @ {fps} FPS")

        # Tính toán tối ưu hóa
        new_width = int(width * self.scale)
        new_height = int(height * self.scale)
        print(f"[INFO] Đang xử lý: {new_width}×{new_height}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Bỏ qua frames
            if frame_count % self.skip_frames != 0:
                if self.last_result is not None:
                    cv2.imshow('Result', self.last_result)
                continue

            # Resize
            small_frame = cv2.resize(frame, (new_width, new_height))

            # Trích xuất ROI nếu được định nghĩa
            if self.roi:
                x, y, w, h = self.roi
                x, y, w, h = int(x * self.scale), int(y * self.scale), int(w * self.scale), int(h * self.scale)
                processing_frame = small_frame[y:y+h, x:x+w]
            else:
                processing_frame = small_frame

            # Xử lý (đã tối ưu)
            result = self._optimized_processing(processing_frame)

            # Resize lại
            result = cv2.resize(result, (width, height))

            self.last_result = result

            # Hiển thị
            cv2.imshow('Result', result)

            # Duy trì target FPS
            elapsed = time.time() - start_time
            expected_time = frame_count / self.target_fps
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Thống kê
        total_time = time.time() - start_time
        actual_fps = frame_count / total_time

        print(f"\n[THỐNG KÊ]")
        print(f"Frames đã xử lý: {frame_count}")
        print(f"Tổng thời gian: {total_time:.2f}s")
        print(f"FPS thực tế: {actual_fps:.1f}")
        print(f"FPS mục tiêu: {self.target_fps}")

        cap.release()
        cv2.destroyAllWindows()

    def _optimized_processing(self, frame):
        """Pipeline xử lý đã tối ưu"""

        # Chuyển sang grayscale (một lần)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur (kernel đã tối ưu)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold (phương pháp nhanh)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphology (tối thiểu)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return binary


# Sử dụng
processor = OptimizedVideoProcessor('video.mp4', target_fps=30)
processor.roi = (400, 200, 800, 600)  # ROI tùy chọn
processor.process()
```

---

## 12. Kết Quả Benchmarking

### Ví Dụ Tối Ưu Hóa

```python
"""
Kết Quả Tối Ưu Hóa cho video 1920×1080:

Cơ sở (không tối ưu hóa):
- FPS: 12
- Thời gian xử lý mỗi frame: 83ms

Với resize (0.5):
- FPS: 25 (+108%)
- Thời gian xử lý mỗi frame: 40ms

Với resize + frame skip (2):
- FPS: 45 (+275%)
- Thời gian xử lý mỗi frame: 22ms

Với resize + frame skip + ROI:
- FPS: 68 (+467%)
- Thời gian xử lý mỗi frame: 15ms

Với tất cả tối ưu hóa + vectorization:
- FPS: 95 (+692%)
- Thời gian xử lý mỗi frame: 11ms
"""
```

---

**Ngày tạo**: Tháng 1/2025
