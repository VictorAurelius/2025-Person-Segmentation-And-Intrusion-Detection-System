# Optimization Techniques

## 1. Profiling and Benchmarking

### A. Timing Code

```python
import time
import cv2

def benchmark_function(func, *args, iterations=100):
    """Benchmark function execution time"""
    times = []

    for _ in range(iterations):
        start = time.time()
        result = func(*args)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"Function: {func.__name__}")
    print(f"  Average: {avg_time * 1000:.2f}ms")
    print(f"  Min: {min_time * 1000:.2f}ms")
    print(f"  Max: {max_time * 1000:.2f}ms")

    return result


# Usage
image = cv2.imread('image.jpg')

def method1(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def method2(img):
    return cv2.blur(img, (5, 5))

benchmark_function(method1, image)
benchmark_function(method2, image)
```

### B. Using timeit

```python
import timeit

# Setup code
setup = """
import cv2
import numpy as np
image = cv2.imread('image.jpg')
"""

# Method 1
code1 = """
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
"""

# Method 2
code2 = """
gray = np.mean(image, axis=2).astype(np.uint8)
"""

time1 = timeit.timeit(code1, setup=setup, number=1000)
time2 = timeit.timeit(code2, setup=setup, number=1000)

print(f"cv2.cvtColor: {time1:.4f}s")
print(f"np.mean: {time2:.4f}s")
print(f"Faster: {'cv2.cvtColor' if time1 < time2 else 'np.mean'} by {abs(time1 - time2) / min(time1, time2) * 100:.1f}%")
```

### C. Line Profiler

```python
# Install: pip install line_profiler

from line_profiler import LineProfiler
import cv2

def process_image(image):
    """Function to profile"""
    # Line 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Line 2
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Line 3
    edges = cv2.Canny(blurred, 50, 150)

    # Line 4
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# Profile
profiler = LineProfiler()
profiler.add_function(process_image)

image = cv2.imread('image.jpg')

# Run with profiler
profiler.enable()
result = process_image(image)
profiler.disable()

# Print stats
profiler.print_stats()
```

---

## 2. Image Resizing

### A. Reduce Resolution

```python
# Original: 1920×1080
image = cv2.imread('image.jpg')
print(f"Original: {image.shape}")

# Resize to 50%
scale = 0.5
small = cv2.resize(image, None, fx=scale, fy=scale)
print(f"Resized: {small.shape}")

# Process small image (4x faster!)
processed = process_frame(small)

# Resize back if needed
large = cv2.resize(processed, (image.shape[1], image.shape[0]))
```

**Speed Comparison:**
```
Resolution    Pixels     Relative Speed
1920×1080    2,073,600   1x (baseline)
1280×720     921,600     2.25x faster
960×540      518,400     4x faster
640×480      307,200     6.75x faster
```

### B. Smart Resizing

```python
def smart_resize(image, target_pixels=500000):
    """Resize to target pixel count"""
    h, w = image.shape[:2]
    current_pixels = h * w

    if current_pixels <= target_pixels:
        return image

    # Calculate scale
    scale = (target_pixels / current_pixels) ** 0.5

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    return resized


# Auto-resize large images
image = cv2.imread('large_image.jpg')
optimized = smart_resize(image, target_pixels=500000)

print(f"Original: {image.shape[1]}×{image.shape[0]}")
print(f"Optimized: {optimized.shape[1]}×{optimized.shape[0]}")
```

---

## 3. ROI Processing

### A. Process Only ROI

```python
# Full frame processing (slow)
def process_full_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


# ROI processing (fast)
def process_roi_only(frame, roi):
    """Process only region of interest"""
    x, y, w, h = roi

    # Extract ROI
    roi_frame = frame[y:y+h, x:x+w]

    # Process ROI
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Create full-size result
    result = np.zeros(frame.shape[:2], dtype=np.uint8)
    result[y:y+h, x:x+w] = edges

    return result


# Benchmark
roi = (400, 200, 800, 600)  # Only process 800×600 region of 1920×1080

time1 = timeit.timeit(lambda: process_full_frame(frame), number=100)
time2 = timeit.timeit(lambda: process_roi_only(frame, roi), number=100)

print(f"Full frame: {time1:.4f}s")
print(f"ROI only: {time2:.4f}s")
print(f"Speedup: {time1 / time2:.2f}x")
```

---

## 4. Frame Skipping

### A. Process Every Nth Frame

```python
class FrameSkipProcessor:
    """Process every Nth frame, reuse results"""

    def __init__(self, skip_frames=2):
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.last_result = None

    def process(self, frame):
        """Process frame or return cached result"""
        self.frame_count += 1

        # Process this frame?
        if self.frame_count % self.skip_frames == 0:
            self.last_result = expensive_processing(frame)

        return self.last_result


# Usage
processor = FrameSkipProcessor(skip_frames=3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 3rd frame
    result = processor.process(frame)

    cv2.imshow('Result', result)
```

**FPS Improvement:**
```
skip_frames = 1:  15 FPS (process all)
skip_frames = 2:  25 FPS (67% faster)
skip_frames = 3:  35 FPS (133% faster)
```

---

## 5. Caching

### A. LRU Cache

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def expensive_computation(image_hash):
    """Cached expensive computation"""
    # Simulate expensive operation
    time.sleep(0.1)
    return image_hash


def process_with_cache(image):
    """Process image with caching"""
    # Create hash
    image_hash = hashlib.md5(image.tobytes()).hexdigest()

    # Use cached result if available
    result = expensive_computation(image_hash)

    return result


# First call: slow (cache miss)
start = time.time()
result1 = process_with_cache(image)
print(f"First call: {time.time() - start:.4f}s")

# Second call: fast (cache hit)
start = time.time()
result2 = process_with_cache(image)
print(f"Second call: {time.time() - start:.4f}s")
```

### B. Result Caching

```python
class ResultCache:
    """Cache processing results"""

    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size

    def get(self, key):
        """Get cached result"""
        return self.cache.get(key)

    def put(self, key, value):
        """Store result"""
        # Simple LRU: remove oldest if full
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        self.cache[key] = value

    def clear(self):
        """Clear cache"""
        self.cache.clear()


# Usage
cache = ResultCache(max_size=100)

def process_frame_cached(frame, frame_number):
    """Process with caching"""

    # Check cache
    cached = cache.get(frame_number)
    if cached is not None:
        return cached

    # Process
    result = expensive_processing(frame)

    # Store
    cache.put(frame_number, result)

    return result
```

---

## 6. NumPy Optimizations

### A. Vectorization

```python
import numpy as np

# Slow: Loop
def brighten_loop(image, value):
    result = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = min(255, image[i, j] + value)
    return result


# Fast: Vectorized
def brighten_vectorized(image, value):
    return np.clip(image + value, 0, 255).astype(np.uint8)


# Benchmark
image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

time1 = timeit.timeit(lambda: brighten_loop(image, 50), number=10)
time2 = timeit.timeit(lambda: brighten_vectorized(image, 50), number=10)

print(f"Loop: {time1:.4f}s")
print(f"Vectorized: {time2:.4f}s")
print(f"Speedup: {time1 / time2:.0f}x")  # ~1000x faster!
```

### B. In-Place Operations

```python
# Creates copy (slower, more memory)
def process_copy(image):
    result = image.copy()
    result = result * 1.5
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


# In-place (faster, less memory)
def process_inplace(image):
    image = image.astype(np.float32)
    image *= 1.5
    np.clip(image, 0, 255, out=image)
    return image.astype(np.uint8)
```

### C. Data Types

```python
# Use appropriate dtype
image_uint8 = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)   # 6.2 MB
image_float32 = image_uint8.astype(np.float32)  # 24.8 MB (4x larger!)
image_float64 = image_uint8.astype(np.float64)  # 49.6 MB (8x larger!)

# Convert only when necessary
def process_smart(image):
    # Process in uint8 when possible
    result = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert to float only for operations that need it
    if need_float_precision:
        result = result.astype(np.float32)
        result *= 1.5
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result
```

---

## 7. OpenCV Optimizations

### A. Use Appropriate Functions

```python
# Slow: Multiple operations
def convert_slow(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    return binary


# Fast: Combined operation
def convert_fast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary
```

### B. Avoid Repeated Conversions

```python
# Inefficient
for _ in range(100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert every time
    process(gray)


# Efficient
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert once
for _ in range(100):
    process(gray)
```

### C. Use Built-in Functions

```python
# Slow: Manual implementation
def manual_blur(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    result = cv2.filter2D(image, -1, kernel)
    return result


# Fast: Built-in function (optimized)
def builtin_blur(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))


# Built-in is 5-10x faster!
```

---

## 8. Memory Management

### A. Preallocate Arrays

```python
# Slow: Create new array each time
def process_slow(num_frames):
    results = []
    for i in range(num_frames):
        result = np.zeros((1080, 1920, 3), dtype=np.uint8)
        results.append(result)
    return results


# Fast: Preallocate
def process_fast(num_frames):
    results = np.zeros((num_frames, 1080, 1920, 3), dtype=np.uint8)
    for i in range(num_frames):
        results[i] = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return results
```

### B. Release Memory

```python
import gc

def process_large_batch(images):
    """Process large batch with memory management"""

    results = []

    for i, image in enumerate(images):
        # Process
        result = expensive_processing(image)
        results.append(result)

        # Periodically release memory
        if i % 100 == 0:
            gc.collect()

    return results
```

---

## 9. GPU Acceleration

### A. Check CUDA Availability

```python
# Check if OpenCV built with CUDA
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
```

### B. GPU Processing

```python
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    # Upload to GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)

    # Process on GPU
    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
    gpu_blurred = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0
    ).apply(gpu_gray)

    # Download result
    result = gpu_blurred.download()

else:
    # Fallback to CPU
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(gray, (5, 5), 0)
```

---

## 10. Algorithm Selection

### A. Choose Right Algorithm

```python
# Scenario: Need to blur image

# Small kernel (3×3): cv2.blur (fastest)
if kernel_size <= 3:
    blurred = cv2.blur(image, (3, 3))

# Medium kernel (5×5 to 11×11): cv2.GaussianBlur (balanced)
elif kernel_size <= 11:
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Large kernel (>11): cv2.boxFilter + multiple passes (faster)
else:
    blurred = image
    for _ in range(3):
        blurred = cv2.boxFilter(blurred, -1, (5, 5))
```

### B. Trade-offs

```python
# Scenario: Motion detection

# Highest accuracy (slowest)
def method_accurate(frame, bg_model):
    return cv2.createBackgroundSubtractorKNN().apply(frame)


# Balanced (recommended)
def method_balanced(frame, bg_model):
    return cv2.createBackgroundSubtractorMOG2().apply(frame)


# Fastest (lower accuracy)
def method_fast(frame, prev_frame):
    diff = cv2.absdiff(frame, prev_frame)
    _, binary = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return binary
```

---

## 11. Complete Optimization Example

```python
class OptimizedVideoProcessor:
    """Highly optimized video processor"""

    def __init__(self, video_path, target_fps=30):
        self.video_path = video_path
        self.target_fps = target_fps

        # Optimizations
        self.scale = 0.5  # Resize to 50%
        self.skip_frames = 2  # Process every 2nd frame
        self.roi = None  # Process specific ROI only

        # Caching
        self.cache = {}
        self.last_result = None

    def process(self):
        """Process video with optimizations"""

        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        start_time = time.time()

        # Get properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] Original: {width}×{height} @ {fps} FPS")

        # Calculate optimizations
        new_width = int(width * self.scale)
        new_height = int(height * self.scale)
        print(f"[INFO] Processing: {new_width}×{new_height}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames
            if frame_count % self.skip_frames != 0:
                if self.last_result is not None:
                    cv2.imshow('Result', self.last_result)
                continue

            # Resize
            small_frame = cv2.resize(frame, (new_width, new_height))

            # Extract ROI if defined
            if self.roi:
                x, y, w, h = self.roi
                x, y, w, h = int(x * self.scale), int(y * self.scale), int(w * self.scale), int(h * self.scale)
                processing_frame = small_frame[y:y+h, x:x+w]
            else:
                processing_frame = small_frame

            # Process (optimized)
            result = self._optimized_processing(processing_frame)

            # Resize back
            result = cv2.resize(result, (width, height))

            self.last_result = result

            # Display
            cv2.imshow('Result', result)

            # Maintain target FPS
            elapsed = time.time() - start_time
            expected_time = frame_count / self.target_fps
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stats
        total_time = time.time() - start_time
        actual_fps = frame_count / total_time

        print(f"\n[STATS]")
        print(f"Frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Actual FPS: {actual_fps:.1f}")
        print(f"Target FPS: {self.target_fps}")

        cap.release()
        cv2.destroyAllWindows()

    def _optimized_processing(self, frame):
        """Optimized processing pipeline"""

        # Convert to grayscale (once)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur (optimized kernel)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold (fast method)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphology (minimal)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return binary


# Usage
processor = OptimizedVideoProcessor('video.mp4', target_fps=30)
processor.roi = (400, 200, 800, 600)  # Optional ROI
processor.process()
```

---

## 12. Benchmarking Results

### Example Optimizations

```python
"""
Optimization Results for 1920×1080 video:

Baseline (no optimizations):
- FPS: 12
- Processing time per frame: 83ms

With resize (0.5):
- FPS: 25 (+108%)
- Processing time per frame: 40ms

With resize + frame skip (2):
- FPS: 45 (+275%)
- Processing time per frame: 22ms

With resize + frame skip + ROI:
- FPS: 68 (+467%)
- Processing time per frame: 15ms

With all optimizations + vectorization:
- FPS: 95 (+692%)
- Processing time per frame: 11ms
"""
```

---

**Ngày tạo**: Tháng 1/2025
