# Frame Differencing (So Sánh Khung Hình)

## 1. Khái Niệm

**Frame Differencing (so sánh khung hình)** là kỹ thuật Motion Detection (phát hiện chuyển động) đơn giản nhất, phát hiện chuyển động bằng cách so sánh sự khác biệt giữa các frames liên tiếp.

### A. Nguyên Lý Cơ Bản

```
Chuyển Động = | Frame(t) - Frame(t-1) |
```

**Nếu pixel thay đổi đáng kể** → Phát hiện chuyển động
**Nếu pixel không đổi** → Nền tĩnh

---

## 2. Two-Frame Differencing (So Sánh Hai Khung Hình)

### A. Triển Khai Đơn Giản

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

# Đọc frame đầu tiên
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tính sự khác biệt tuyệt đối
    diff = cv2.absdiff(prev_gray, gray)

    # Ngưỡng hóa để lấy binary mask
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Hiển thị
    cv2.imshow('Motion Mask', motion_mask)
    cv2.imshow('Original', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Cập nhật frame trước
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
```

### B. Với Hậu Xử Lý

```python
def two_frame_differencing(prev_frame, curr_frame, threshold=25):
    """Two-frame differencing với hậu xử lý"""

    # Chuyển sang ảnh xám
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Làm mờ để giảm nhiễu
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # Tính sự khác biệt
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Ngưỡng hóa
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Phép toán hình thái
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Loại bỏ nhiễu (opening)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

    # Lấp lỗ (closing)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    return motion_mask
```

---

## 3. Three-Frame Differencing (So Sánh Ba Khung Hình)

### A. Nguyên Lý

Three-frame differencing giảm false positives (phát hiện sai) bằng cách yêu cầu chuyển động xuất hiện trong cả 2 sự khác biệt:

```
Diff1 = | Frame(t) - Frame(t-1) |
Diff2 = | Frame(t) - Frame(t-2) |
Chuyển Động = Diff1 AND Diff2
```

### B. Triển Khai

```python
class ThreeFrameDifferencing:
    """Three-frame differencing cho phát hiện chuyển động mạnh mẽ"""

    def __init__(self, threshold=25):
        self.threshold = threshold
        self.frame_buffer = []

    def detect(self, frame):
        """Phát hiện chuyển động sử dụng ba frames"""

        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Làm mờ để giảm nhiễu
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thêm vào buffer
        self.frame_buffer.append(gray)

        # Chỉ giữ 3 frames
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        # Cần 3 frames để xử lý
        if len(self.frame_buffer) < 3:
            return np.zeros(gray.shape, dtype=np.uint8)

        # Lấy frames
        frame1, frame2, frame3 = self.frame_buffer

        # Tính sự khác biệt
        diff1 = cv2.absdiff(frame2, frame1)
        diff2 = cv2.absdiff(frame3, frame2)

        # Ngưỡng hóa
        _, mask1 = cv2.threshold(diff1, self.threshold, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(diff2, self.threshold, 255, cv2.THRESH_BINARY)

        # Phép AND logic
        motion_mask = cv2.bitwise_and(mask1, mask2)

        # Phép toán hình thái
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        return motion_mask


# Sử dụng
detector = ThreeFrameDifferencing(threshold=25)

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện chuyển động
    motion_mask = detector.detect(frame)

    # Hiển thị
    cv2.imshow('Motion', motion_mask)
    cv2.imshow('Original', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### C. Ưu Điểm Của Three-Frame

**Vấn Đề Two-Frame:**
```
Frame1: [Vật thể tại vị trí A]
Frame2: [Vật thể di chuyển đến vị trí B]
Kết quả: Phát hiện chuyển động tại cả A và B (false positive tại A)
```

**Giải Pháp Three-Frame:**
```
Frame1: [Vật thể tại vị trí A]
Frame2: [Vật thể di chuyển đến vị trí B]
Frame3: [Vật thể tại vị trí C]

Diff1 = B và A (chuyển động tại cả hai)
Diff2 = C và B (chuyển động chỉ tại B và C)
AND = Chỉ B (đúng!)
```

---

## 4. Weighted Frame Differencing (So Sánh Khung Hình Có Trọng Số)

### A. Exponential Moving Average (Trung Bình Động Mũ)

```python
class WeightedFrameDiff:
    """Frame differencing với lịch sử có trọng số"""

    def __init__(self, alpha=0.1, threshold=25):
        self.alpha = alpha  # Tốc độ học
        self.threshold = threshold
        self.avg = None

    def detect(self, frame):
        """Phát hiện chuyển động sử dụng trung bình có trọng số"""

        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_float = gray.astype(np.float32)

        # Khởi tạo trung bình
        if self.avg is None:
            self.avg = gray_float.copy()
            return np.zeros(gray.shape, dtype=np.uint8)

        # Cập nhật trung bình có trọng số
        cv2.accumulateWeighted(gray_float, self.avg, self.alpha)

        # Tính sự khác biệt
        diff = cv2.absdiff(gray_float, self.avg)
        diff = diff.astype(np.uint8)

        # Ngưỡng hóa
        _, motion_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Phép toán hình thái
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        return motion_mask


# Sử dụng
detector = WeightedFrameDiff(alpha=0.05, threshold=20)
```

**Tham Số Alpha:**
```
alpha = 0.01:  Thích ứng rất chậm (nền ổn định)
alpha = 0.05:  Thích ứng chậm (khuyến nghị)
alpha = 0.1:   Thích ứng trung bình
alpha = 0.5:   Thích ứng nhanh (theo dõi thay đổi nhanh)
```

---

## 5. Adaptive Thresholding (Ngưỡng Hóa Thích Ứng)

### A. Chọn Ngưỡng Tự Động

```python
def auto_threshold_diff(prev_gray, curr_gray, percentile=95):
    """Tự động chọn ngưỡng dựa trên phân phối sự khác biệt"""

    # Tính sự khác biệt
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Dùng percentile làm ngưỡng
    threshold = np.percentile(diff, percentile)

    # Áp dụng ngưỡng
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return motion_mask, threshold
```

### B. Otsu's Threshold (Ngưỡng Otsu)

```python
def otsu_frame_diff(prev_gray, curr_gray):
    """Sử dụng phương pháp Otsu cho ngưỡng tự động"""

    # Tính sự khác biệt
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Ngưỡng Otsu
    threshold, motion_mask = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"Ngưỡng tự động: {threshold}")

    return motion_mask
```

---

## 6. Region-Based Differencing (So Sánh Theo Vùng)

### A. Block-Based Motion Detection (Phát Hiện Chuyển Động Theo Khối)

```python
def block_based_motion(prev_gray, curr_gray, block_size=16, threshold=500):
    """Phát hiện chuyển động ở mức khối"""

    h, w = prev_gray.shape
    motion_map = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            # Trích xuất khối
            block_prev = prev_gray[y:y+block_size, x:x+block_size]
            block_curr = curr_gray[y:y+block_size, x:x+block_size]

            # Tính sự khác biệt
            diff = cv2.absdiff(block_prev, block_curr)
            block_motion = np.sum(diff)

            # Nếu chuyển động vượt ngưỡng
            if block_motion > threshold:
                motion_map[y:y+block_size, x:x+block_size] = 255

    return motion_map
```

### B. Trực Quan Hóa Lưới

```python
def visualize_block_motion(frame, motion_map, block_size=16):
    """Trực quan hóa chuyển động theo khối"""

    result = frame.copy()
    h, w = motion_map.shape

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            if motion_map[y, x] == 255:
                # Vẽ hình chữ nhật cho các khối chuyển động
                cv2.rectangle(result, (x, y), (x + block_size, y + block_size),
                            (0, 255, 0), 2)

    return result
```

---

## 7. Bộ Phát Hiện Chuyển Động Hoàn Chỉnh

```python
class FrameDifferencingDetector:
    """Bộ phát hiện chuyển động frame differencing hoàn chỉnh"""

    def __init__(self, method='three_frame', threshold=25, min_area=500):
        self.method = method
        self.threshold = threshold
        self.min_area = min_area
        self.frame_buffer = []

    def detect(self, frame):
        """Phát hiện chuyển động trong frame"""

        # Chuyển sang ảnh xám và làm mờ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thêm vào buffer
        self.frame_buffer.append(gray)

        # Giới hạn kích thước buffer
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        # Chọn phương pháp
        if self.method == 'two_frame':
            motion_mask = self._two_frame()
        elif self.method == 'three_frame':
            motion_mask = self._three_frame()
        else:
            motion_mask = np.zeros(gray.shape, dtype=np.uint8)

        # Hậu xử lý
        motion_mask = self._post_process(motion_mask)

        return motion_mask

    def _two_frame(self):
        """Two-frame differencing"""
        if len(self.frame_buffer) < 2:
            return np.zeros(self.frame_buffer[0].shape, dtype=np.uint8)

        diff = cv2.absdiff(self.frame_buffer[-2], self.frame_buffer[-1])
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        return mask

    def _three_frame(self):
        """Three-frame differencing"""
        if len(self.frame_buffer) < 3:
            return np.zeros(self.frame_buffer[0].shape, dtype=np.uint8)

        # Two differences
        diff1 = cv2.absdiff(self.frame_buffer[-3], self.frame_buffer[-2])
        diff2 = cv2.absdiff(self.frame_buffer[-2], self.frame_buffer[-1])

        # Threshold
        _, mask1 = cv2.threshold(diff1, self.threshold, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(diff2, self.threshold, 255, cv2.THRESH_BINARY)

        # AND operation
        mask = cv2.bitwise_and(mask1, mask2)
        return mask

    def _post_process(self, mask):
        """Post-process mask"""
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    def get_contours(self, mask):
        """Get contours from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc theo diện tích
        filtered = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        return filtered


# Sử dụng
detector = FrameDifferencingDetector(method='three_frame', threshold=25, min_area=500)

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện chuyển động
    motion_mask = detector.detect(frame)

    # Lấy contours
    contours = detector.get_contours(motion_mask)

    # Vẽ kết quả
    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Hiển thị
    cv2.imshow('Motion Mask', motion_mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 8. So Sánh Với Các Phương Pháp Khác

### A. Frame Differencing vs Background Subtraction (So Sánh)

| Aspect | Frame Diff | Background Sub |
|--------|------------|----------------|
| Speed | Very Fast ✅ | Slower |
| Accuracy | Lower | Higher ✅ |
| Adaptation | N/A | Yes ✅ |
| Setup | None ✅ | Learning phase |
| Static Objects | Not detected ✅ | May detect ⚠️ |
| Lighting Changes | Sensitive ⚠️ | Adaptive ✅ |

### B. Khi Nào Dùng Frame Differencing

**Tốt cho:**
- ✅ Fast moving objects
- ✅ Real-time performance critical
- ✅ Simple scenes
- ✅ Temporary motion detection
- ✅ Quick prototyping

**Không tốt cho:**
- ❌ Slow moving objects
- ❌ Static camera with learning phase available
- ❌ Complex lighting conditions
- ❌ Long-term monitoring

---

## 9. Tối Ưu Hiệu Năng

### A. Xử Lý Đa Tỷ Lệ

```python
def multi_scale_diff(prev_frame, curr_frame, scales=[1.0, 0.5, 0.25]):
    """Multi-scale frame differencing"""

    motion_masks = []

    for scale in scales:
        # Resize
        if scale != 1.0:
            prev_scaled = cv2.resize(prev_frame, None, fx=scale, fy=scale)
            curr_scaled = cv2.resize(curr_frame, None, fx=scale, fy=scale)
        else:
            prev_scaled = prev_frame
            curr_scaled = curr_frame

        # Phát hiện chuyển động
        diff = cv2.absdiff(prev_scaled, curr_scaled)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Resize back
        if scale != 1.0:
            mask = cv2.resize(mask, (prev_frame.shape[1], prev_frame.shape[0]))

        motion_masks.append(mask)

    # Combine masks (logical OR)
    combined = motion_masks[0]
    for mask in motion_masks[1:]:
        combined = cv2.bitwise_or(combined, mask)

    return combined
```

### B. Tăng Tốc GPU

```python
import cv2.cuda as cuda

# Upload to GPU
gpu_frame1 = cuda.GpuMat()
gpu_frame2 = cuda.GpuMat()

gpu_frame1.upload(prev_gray)
gpu_frame2.upload(curr_gray)

# Calculate difference on GPU
gpu_diff = cuda.absdiff(gpu_frame1, gpu_frame2)

# Threshold on GPU
_, gpu_mask = cuda.threshold(gpu_diff, 25, 255, cv2.THRESH_BINARY)

# Download result
motion_mask = gpu_mask.download()
```

---

## 10. Khắc Phục Sự Cố

### A. Quá Nhiều False Positives

**Nguyên nhân:**
- Camera shake
- Lighting changes
- Noise

**Giải pháp:**
```python
# 1. Increase threshold
detector = FrameDifferencingDetector(threshold=35)

# 2. More aggressive morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 3. Larger minimum area
detector = FrameDifferencingDetector(min_area=1000)

# 4. Use three-frame differencing
detector = FrameDifferencingDetector(method='three_frame')

# 5. Stabilize camera (if possible)
```

### B. Bỏ Lỡ Chuyển Động Chậm

**Nguyên nhân:**
- Object moves too slowly between frames
- Threshold too high

**Giải pháp:**
```python
# 1. Lower threshold
detector = FrameDifferencingDetector(threshold=15)

# 2. Use weighted differencing (accumulates small changes)
detector = WeightedFrameDiff(alpha=0.05, threshold=20)

# 3. Compare with older frames (not just previous)
diff = cv2.absdiff(frame_buffer[-1], frame_buffer[-5])  # 5 frames ago
```

### C. Vật Thể Ảo

**Vấn đề:** Objects leave "trails" or appear at old positions

**Nguyên nhân:** Two-frame differencing creates ghosts

**Giải pháp:**
```python
# Use three-frame differencing (eliminates ghosts)
detector = FrameDifferencingDetector(method='three_frame')
```

**Trực quan:**
```
Two-Frame:
Frame t-1: [Object at A]
Frame t:   [Object at B]
Result:    Motion at A and B (ghost at A!)

Three-Frame:
Frame t-2: [Object at A]
Frame t-1: [Object at A.5]
Frame t:   [Object at B]
Diff1: Motion at A and A.5
Diff2: Motion at A.5 and B
AND:   Motion at A.5 only (no ghost!)
```

---

## 11. Kỹ Thuật Nâng Cao

### A. Bộ Lọc Trung Vị Thời Gian

```python
from collections import deque

class TemporalMedianDiff:
    """Frame differencing with temporal median filtering"""

    def __init__(self, buffer_size=5, threshold=25):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.buffer = deque(maxlen=buffer_size)

    def detect(self, frame):
        """Detect motion using temporal median"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self.buffer.append(gray)

        if len(self.buffer) < self.buffer_size:
            return np.zeros(gray.shape, dtype=np.uint8)

        # Calculate temporal median
        frames_stack = np.array(self.buffer)
        median_frame = np.median(frames_stack, axis=0).astype(np.uint8)

        # Difference with median
        diff = cv2.absdiff(gray, median_frame)

        # Threshold
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        return mask
```

### B. Sự Khác Biệt Frame Song Phương

```python
def bilateral_frame_diff(prev_gray, curr_gray, threshold=25):
    """Frame differencing with bilateral filtering"""

    # Apply bilateral filter (preserve edges while smoothing)
    prev_filtered = cv2.bilateralFilter(prev_gray, 9, 75, 75)
    curr_filtered = cv2.bilateralFilter(curr_gray, 9, 75, 75)

    # Calculate difference
    diff = cv2.absdiff(prev_filtered, curr_filtered)

    # Threshold
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return mask
```

---

**Ngày tạo**: Tháng 1/2025
