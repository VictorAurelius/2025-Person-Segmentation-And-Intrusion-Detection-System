# Frame Differencing

## 1. Khái Niệm

**Frame Differencing** là kỹ thuật motion detection đơn giản nhất, phát hiện chuyển động bằng cách so sánh sự khác biệt giữa các frames liên tiếp.

### A. Nguyên Lý Cơ Bản

```
Motion = | Frame(t) - Frame(t-1) |
```

**Nếu pixel thay đổi đáng kể** → Motion detected
**Nếu pixel không đổi** → Static background

---

## 2. Two-Frame Differencing

### A. Simple Implementation

```python
import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

# Read first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference
    diff = cv2.absdiff(prev_gray, gray)

    # Threshold to get binary mask
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Display
    cv2.imshow('Motion Mask', motion_mask)
    cv2.imshow('Original', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update previous frame
    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
```

### B. Với Post-Processing

```python
def two_frame_differencing(prev_frame, curr_frame, threshold=25):
    """Two-frame differencing với post-processing"""

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    # Calculate difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Threshold
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Remove noise (opening)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

    # Fill holes (closing)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    return motion_mask
```

---

## 3. Three-Frame Differencing

### A. Nguyên Lý

Three-frame differencing giảm false positives bằng cách yêu cầu motion xuất hiện trong cả 2 differences:

```
Diff1 = | Frame(t) - Frame(t-1) |
Diff2 = | Frame(t) - Frame(t-2) |
Motion = Diff1 AND Diff2
```

### B. Implementation

```python
class ThreeFrameDifferencing:
    """Three-frame differencing for robust motion detection"""

    def __init__(self, threshold=25):
        self.threshold = threshold
        self.frame_buffer = []

    def detect(self, frame):
        """Detect motion using three frames"""

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Add to buffer
        self.frame_buffer.append(gray)

        # Keep only 3 frames
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        # Need 3 frames for processing
        if len(self.frame_buffer) < 3:
            return np.zeros(gray.shape, dtype=np.uint8)

        # Get frames
        frame1, frame2, frame3 = self.frame_buffer

        # Calculate differences
        diff1 = cv2.absdiff(frame2, frame1)
        diff2 = cv2.absdiff(frame3, frame2)

        # Threshold
        _, mask1 = cv2.threshold(diff1, self.threshold, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(diff2, self.threshold, 255, cv2.THRESH_BINARY)

        # Logical AND
        motion_mask = cv2.bitwise_and(mask1, mask2)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        return motion_mask


# Usage
detector = ThreeFrameDifferencing(threshold=25)

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect motion
    motion_mask = detector.detect(frame)

    # Display
    cv2.imshow('Motion', motion_mask)
    cv2.imshow('Original', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### C. Advantages of Three-Frame

**Two-Frame Issues:**
```
Frame1: [Object at position A]
Frame2: [Object moved to position B]
Result: Motion detected at both A and B (false positive at A)
```

**Three-Frame Solution:**
```
Frame1: [Object at position A]
Frame2: [Object moved to position B]
Frame3: [Object at position C]

Diff1 = B and A (motion at both)
Diff2 = C and B (motion only at B and C)
AND = Only B (correct!)
```

---

## 4. Weighted Frame Differencing

### A. Exponential Moving Average

```python
class WeightedFrameDiff:
    """Frame differencing with weighted history"""

    def __init__(self, alpha=0.1, threshold=25):
        self.alpha = alpha  # Learning rate
        self.threshold = threshold
        self.avg = None

    def detect(self, frame):
        """Detect motion using weighted average"""

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_float = gray.astype(np.float32)

        # Initialize average
        if self.avg is None:
            self.avg = gray_float.copy()
            return np.zeros(gray.shape, dtype=np.uint8)

        # Update weighted average
        cv2.accumulateWeighted(gray_float, self.avg, self.alpha)

        # Calculate difference
        diff = cv2.absdiff(gray_float, self.avg)
        diff = diff.astype(np.uint8)

        # Threshold
        _, motion_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        return motion_mask


# Usage
detector = WeightedFrameDiff(alpha=0.05, threshold=20)
```

**Alpha Parameter:**
```
alpha = 0.01:  Very slow adaptation (stable background)
alpha = 0.05:  Slow adaptation (recommended)
alpha = 0.1:   Medium adaptation
alpha = 0.5:   Fast adaptation (tracks changes quickly)
```

---

## 5. Adaptive Thresholding

### A. Auto-Threshold Selection

```python
def auto_threshold_diff(prev_gray, curr_gray, percentile=95):
    """Auto-select threshold based on difference distribution"""

    # Calculate difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Use percentile as threshold
    threshold = np.percentile(diff, percentile)

    # Apply threshold
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return motion_mask, threshold
```

### B. Otsu's Threshold

```python
def otsu_frame_diff(prev_gray, curr_gray):
    """Use Otsu's method for automatic threshold"""

    # Calculate difference
    diff = cv2.absdiff(prev_gray, curr_gray)

    # Otsu threshold
    threshold, motion_mask = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print(f"Auto threshold: {threshold}")

    return motion_mask
```

---

## 6. Region-Based Differencing

### A. Block-Based Motion Detection

```python
def block_based_motion(prev_gray, curr_gray, block_size=16, threshold=500):
    """Detect motion at block level"""

    h, w = prev_gray.shape
    motion_map = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            # Extract blocks
            block_prev = prev_gray[y:y+block_size, x:x+block_size]
            block_curr = curr_gray[y:y+block_size, x:x+block_size]

            # Calculate difference
            diff = cv2.absdiff(block_prev, block_curr)
            block_motion = np.sum(diff)

            # If motion exceeds threshold
            if block_motion > threshold:
                motion_map[y:y+block_size, x:x+block_size] = 255

    return motion_map
```

### B. Grid Visualization

```python
def visualize_block_motion(frame, motion_map, block_size=16):
    """Visualize block-based motion"""

    result = frame.copy()
    h, w = motion_map.shape

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            if motion_map[y, x] == 255:
                # Draw rectangle for motion blocks
                cv2.rectangle(result, (x, y), (x + block_size, y + block_size),
                            (0, 255, 0), 2)

    return result
```

---

## 7. Complete Motion Detector

```python
class FrameDifferencingDetector:
    """Complete frame differencing motion detector"""

    def __init__(self, method='three_frame', threshold=25, min_area=500):
        self.method = method
        self.threshold = threshold
        self.min_area = min_area
        self.frame_buffer = []

    def detect(self, frame):
        """Detect motion in frame"""

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Add to buffer
        self.frame_buffer.append(gray)

        # Limit buffer size
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        # Choose method
        if self.method == 'two_frame':
            motion_mask = self._two_frame()
        elif self.method == 'three_frame':
            motion_mask = self._three_frame()
        else:
            motion_mask = np.zeros(gray.shape, dtype=np.uint8)

        # Post-process
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

        # Filter by area
        filtered = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        return filtered


# Usage
detector = FrameDifferencingDetector(method='three_frame', threshold=25, min_area=500)

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect motion
    motion_mask = detector.detect(frame)

    # Get contours
    contours = detector.get_contours(motion_mask)

    # Draw results
    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Display
    cv2.imshow('Motion Mask', motion_mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 8. Comparison với Other Methods

### A. Frame Differencing vs Background Subtraction

| Aspect | Frame Diff | Background Sub |
|--------|------------|----------------|
| Speed | Very Fast ✅ | Slower |
| Accuracy | Lower | Higher ✅ |
| Adaptation | N/A | Yes ✅ |
| Setup | None ✅ | Learning phase |
| Static Objects | Not detected ✅ | May detect ⚠️ |
| Lighting Changes | Sensitive ⚠️ | Adaptive ✅ |

### B. When to Use Frame Differencing

**Good for:**
- ✅ Fast moving objects
- ✅ Real-time performance critical
- ✅ Simple scenes
- ✅ Temporary motion detection
- ✅ Quick prototyping

**Not good for:**
- ❌ Slow moving objects
- ❌ Static camera with learning phase available
- ❌ Complex lighting conditions
- ❌ Long-term monitoring

---

## 9. Performance Optimization

### A. Multi-Scale Processing

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

        # Detect motion
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

### B. GPU Acceleration

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

## 10. Troubleshooting

### A. Too Many False Positives

**Causes:**
- Camera shake
- Lighting changes
- Noise

**Solutions:**
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

### B. Missing Slow Motion

**Causes:**
- Object moves too slowly between frames
- Threshold too high

**Solutions:**
```python
# 1. Lower threshold
detector = FrameDifferencingDetector(threshold=15)

# 2. Use weighted differencing (accumulates small changes)
detector = WeightedFrameDiff(alpha=0.05, threshold=20)

# 3. Compare with older frames (not just previous)
diff = cv2.absdiff(frame_buffer[-1], frame_buffer[-5])  # 5 frames ago
```

### C. Ghost Objects

**Problem:** Objects leave "trails" or appear at old positions

**Cause:** Two-frame differencing creates ghosts

**Solution:**
```python
# Use three-frame differencing (eliminates ghosts)
detector = FrameDifferencingDetector(method='three_frame')
```

**Visualization:**
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

## 11. Advanced Techniques

### A. Temporal Median Filter

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

### B. Bilateral Frame Difference

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
