# Background Subtraction

## 1. Khái Niệm

**Background Subtraction** là kỹ thuật phân tách foreground (đối tượng chuyển động) khỏi background (nền tĩnh) bằng cách modeling background và so sánh với frame hiện tại.

### A. Nguyên Lý Cơ Bản

```
Current Frame - Background Model = Foreground Mask
```

**Ví dụ:**
```python
# Simple background subtraction
background = first_frame.copy()

while True:
    ret, frame = cap.read()

    # Subtract
    diff = cv2.absdiff(frame, background)

    # Threshold to get mask
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
```

---

## 2. MOG2 (Mixture of Gaussians 2)

### A. Giới Thiệu

MOG2 là adaptive background subtraction algorithm dựa trên Gaussian Mixture Model. Mỗi pixel được model bằng mixture của K Gaussian distributions.

**Paper:** "Improved adaptive Gaussian mixture model for background subtraction" - Zivkovic (2004)

### B. Cách Hoạt Động

#### Pixel Modeling

Mỗi pixel `I(x,y)` được model bằng:

```
P(I(x,y)) = Σ(k=1 to K) w_k * N(μ_k, σ_k²)
```

Trong đó:
- `w_k`: Weight của Gaussian thứ k
- `μ_k`: Mean
- `σ_k²`: Variance
- `K`: Số Gaussians (thường 3-5)

#### Classification

```python
# Pseudo-code
if pixel matches any background Gaussian:
    pixel = BACKGROUND
else:
    pixel = FOREGROUND
```

### C. Implementation

```python
# Create MOG2 subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,           # Số frames học background
    varThreshold=16,       # Threshold phát hiện
    detectShadows=True     # Phát hiện shadows
)

# Process video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame, learningRate=-1)  # -1 = auto

    # Post-process
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Foreground', fg_mask)
```

### D. Parameters

#### history (default: 500)

```python
# Short history: Adapt quickly to changes
bg_sub = cv2.createBackgroundSubtractorMOG2(history=200)

# Long history: More stable, slower adaptation
bg_sub = cv2.createBackgroundSubtractorMOG2(history=1000)
```

**Guidelines:**
```
Fast changing scenes: 200-300
Normal scenes: 500-700
Stable scenes: 800-1000
```

#### varThreshold (default: 16)

```python
# Low threshold: More sensitive (more false positives)
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=10)

# High threshold: Less sensitive (may miss slow motion)
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=25)
```

**Effect:**
```
varThreshold = 10:  Sensitivity ████████████ (High)
varThreshold = 16:  Sensitivity ████████ (Medium)
varThreshold = 25:  Sensitivity █████ (Low)
```

#### detectShadows (default: True)

```python
# Enable shadow detection
bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# fg_mask values:
# 0   = Background (black)
# 127 = Shadow (gray)
# 255 = Foreground (white)

# Remove shadows
fg_mask[fg_mask == 127] = 0  # Treat shadows as background
```

### E. Advanced Usage

#### Learning Rate

```python
# Auto learning rate (default)
fg_mask = bg_sub.apply(frame, learningRate=-1)

# Manual learning rate (0.0 to 1.0)
fg_mask = bg_sub.apply(frame, learningRate=0.01)

# No learning (use current model only)
fg_mask = bg_sub.apply(frame, learningRate=0)
```

**Learning Rate Effect:**
```
0.0:  No update (frozen model)
0.001: Very slow adaptation
0.01:  Slow adaptation (recommended for stable scenes)
0.1:   Fast adaptation (for changing scenes)
1.0:   Immediate update (like frame differencing)
```

#### Get/Set Background

```python
# Get learned background
background = bg_sub.getBackgroundImage()
cv2.imshow('Learned Background', background)

# Save background model (for reuse)
# Note: MOG2 doesn't support direct save, need workaround
```

---

## 3. KNN (K-Nearest Neighbors)

### A. Giới Thiệu

KNN background subtractor sử dụng K-nearest neighbors để classify mỗi pixel là background hay foreground.

**Paper:** "Efficient Adaptive Density Estimation per Image Pixel" - Zivkovic & Heijden (2006)

### B. Cách Hoạt Động

```
1. Maintain sample set cho mỗi pixel
2. Compare current pixel với K nearest samples
3. If distance < threshold → Background
4. Else → Foreground
```

### C. Implementation

```python
# Create KNN subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)

# Usage (same as MOG2)
fg_mask = bg_subtractor.apply(frame)
```

### D. Parameters

#### dist2Threshold (default: 400.0)

```python
# Low threshold: More sensitive
bg_sub = cv2.createBackgroundSubtractorKNN(dist2Threshold=200.0)

# High threshold: Less sensitive
bg_sub = cv2.createBackgroundSubtractorKNN(dist2Threshold=600.0)
```

### E. KNN vs MOG2

| Aspect | MOG2 | KNN |
|--------|------|-----|
| Speed | Faster ✅ | Slower ⚠️ |
| Accuracy | Good | Better ✅ |
| Noise Handling | Moderate | Better ✅ |
| Memory | Lower ✅ | Higher ⚠️ |
| Shadow Detection | Yes | Yes |

**Recommendation:**
- Use MOG2 for general purpose (best balance)
- Use KNN when dealing with noisy videos
- Use MOG2 if speed is critical

---

## 4. Post-Processing

### A. Morphological Operations

```python
def post_process_mask(fg_mask):
    """Clean up foreground mask"""
    # Remove noise (opening)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

    # Fill holes (closing)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    # Dilate to connect nearby regions
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=1)

    return fg_mask
```

### B. Shadow Removal

```python
def remove_shadows(fg_mask):
    """Remove shadow pixels (value 127)"""
    # Treat shadows as background
    fg_mask[fg_mask == 127] = 0

    # Or only keep foreground
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    return fg_mask
```

### C. Size Filtering

```python
def filter_by_size(fg_mask, min_area=500):
    """Remove small blobs"""
    # Find contours
    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create clean mask
    clean_mask = np.zeros_like(fg_mask)

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    return clean_mask
```

---

## 5. Complete Example

```python
class MotionDetector:
    """Motion detector using background subtraction"""

    def __init__(self, method='MOG2', history=500, threshold=16):
        self.method = method

        if method == 'MOG2':
            self.bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=threshold,
                detectShadows=True
            )
        elif method == 'KNN':
            self.bg_sub = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=threshold * 25,  # Convert to KNN scale
                detectShadows=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def detect(self, frame):
        """Detect motion in frame"""
        # Apply background subtraction
        fg_mask = self.bg_sub.apply(frame)

        # Remove shadows
        fg_mask[fg_mask == 127] = 0

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        return fg_mask

    def get_contours(self, fg_mask, min_area=500):
        """Find contours from foreground mask"""
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by area
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

        return filtered

    def get_background(self):
        """Get learned background image"""
        return self.bg_sub.getBackgroundImage()


# Usage
detector = MotionDetector(method='MOG2', history=500, threshold=16)

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect motion
    fg_mask = detector.detect(frame)

    # Get contours
    contours = detector.get_contours(fg_mask, min_area=500)

    # Draw results
    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Display
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 6. Troubleshooting

### A. Too Many False Positives

**Problem:** Everything detected as motion

**Solutions:**
```python
# 1. Increase threshold
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=25)

# 2. Longer history
bg_sub = cv2.createBackgroundSubtractorMOG2(history=1000)

# 3. Stronger morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 4. Higher min area
contours = filter_by_area(contours, min_area=1000)
```

### B. Missing Slow Motion

**Problem:** Slow-moving objects not detected

**Solutions:**
```python
# 1. Lower threshold
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=10)

# 2. Slower learning
fg_mask = bg_sub.apply(frame, learningRate=0.001)

# 3. Use KNN (better for slow motion)
bg_sub = cv2.createBackgroundSubtractorKNN()
```

### C. Background Becomes Foreground

**Problem:** Static objects treated as foreground

**Solutions:**
```python
# 1. Longer history
bg_sub = cv2.createBackgroundSubtractorMOG2(history=1000)

# 2. Let model adapt
for _ in range(100):  # Learn from 100 frames
    bg_sub.apply(frame)

# 3. Manual reset periodically
if frame_count % 1000 == 0:
    bg_sub = cv2.createBackgroundSubtractorMOG2()  # Reset
```

### D. Shadows Detected as Objects

**Problem:** Shadows causing false detections

**Solutions:**
```python
# 1. Enable shadow detection
bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# 2. Remove shadows from mask
fg_mask[fg_mask == 127] = 0

# 3. Adjust shadow threshold
bg_sub.setShadowThreshold(0.5)  # Default is 0.5
```

---

## 7. Performance Tips

### A. Reduce Resolution

```python
# Resize before processing
scale = 0.5
small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
fg_mask = bg_sub.apply(small_frame)

# Resize mask back if needed
fg_mask = cv2.resize(fg_mask, (frame.shape[1], frame.shape[0]))
```

### B. Process ROI Only

```python
# Define ROI
x, y, w, h = 100, 100, 400, 300
roi = frame[y:y+h, x:x+w]

# Process ROI
fg_mask_roi = bg_sub.apply(roi)

# Full mask
fg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
fg_mask[y:y+h, x:x+w] = fg_mask_roi
```

### C. Skip Frames

```python
frame_count = 0
process_every_n = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % process_every_n == 0:
        fg_mask = bg_sub.apply(frame)
        # Process mask...
```

---

## 8. Advanced Techniques

### A. Multiple Background Models

```python
# Use different models for different time periods
class AdaptiveDetector:
    def __init__(self):
        self.day_model = cv2.createBackgroundSubtractorMOG2(history=500)
        self.night_model = cv2.createBackgroundSubtractorMOG2(history=300)

    def detect(self, frame):
        # Determine lighting
        brightness = np.mean(frame)

        if brightness > 100:  # Daytime
            return self.day_model.apply(frame)
        else:  # Nighttime
            return self.night_model.apply(frame)
```

### B. Selective Learning

```python
# Only update background in certain regions
def selective_update(bg_sub, frame, static_regions_mask):
    """Update background only in static regions"""
    # Get foreground with no learning
    fg_mask = bg_sub.apply(frame, learningRate=0)

    # Create learning mask (where to update)
    learn_mask = static_regions_mask & ~fg_mask

    # Update only in learn regions
    fg_mask = bg_sub.apply(frame, learningRate=0.01)

    return fg_mask
```

---

**Ngày tạo**: Tháng 1/2025
