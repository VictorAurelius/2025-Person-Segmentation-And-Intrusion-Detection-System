# Thresholding Techniques

## 1. Khái Niệm

**Thresholding** là kỹ thuật segmentation đơn giản nhất, phân chia image thành foreground và background dựa trên pixel intensity.

### A. Basic Principle

```python
if pixel_value > threshold:
    output = 255  # Foreground (white)
else:
    output = 0    # Background (black)
```

---

## 2. Global Thresholding

### A. Binary Threshold

```python
import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply binary threshold
threshold_value = 127
_, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

cv2.imshow('Original', image)
cv2.imshow('Binary', binary)
cv2.waitKey(0)
```

**Threshold Types:**

```python
# THRESH_BINARY: > threshold → 255, else → 0
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# THRESH_BINARY_INV: > threshold → 0, else → 255
_, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# THRESH_TRUNC: > threshold → threshold, else → same
_, trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

# THRESH_TOZERO: > threshold → same, else → 0
_, tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

# THRESH_TOZERO_INV: > threshold → 0, else → same
_, tozero_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
```

### B. Với Trackbar (Interactive)

```python
def nothing(x):
    pass

cv2.namedWindow('Threshold')
cv2.createTrackbar('Threshold', 'Threshold', 127, 255, nothing)

while True:
    # Get trackbar position
    threshold_value = cv2.getTrackbarPos('Threshold', 'Threshold')

    # Apply threshold
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imshow('Threshold', binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

---

## 3. Otsu's Thresholding

### A. Automatic Threshold Selection

**Otsu's method** tự động tìm optimal threshold bằng cách maximizing inter-class variance.

```python
# Otsu's thresholding
threshold_value, binary = cv2.threshold(
    gray,
    0,        # Threshold value (ignored, auto-calculated)
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"Otsu's threshold: {threshold_value}")
```

### B. Với Gaussian Blur

```python
# Blur before Otsu (recommended)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
threshold_value, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

### C. Complete Example

```python
def otsu_segmentation(image):
    """Segment image using Otsu's method"""

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Blur
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    # Otsu threshold
    threshold_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphology (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary, threshold_value
```

---

## 4. Adaptive Thresholding

### A. Motivation

**Problem với Global Threshold:**
- Ánh sáng không đồng đều
- Shadows
- Gradients

**Solution:** Local thresholds khác nhau cho từng region

### B. Adaptive Mean

```python
# Adaptive threshold với mean
binary = cv2.adaptiveThreshold(
    gray,
    255,                        # Max value
    cv2.ADAPTIVE_THRESH_MEAN_C, # Method
    cv2.THRESH_BINARY,          # Type
    11,                         # Block size (must be odd)
    2                           # Constant C
)
```

**Formula:**
```
threshold(x, y) = mean(neighborhood) - C
```

### C. Adaptive Gaussian

```python
# Adaptive threshold với Gaussian weighted mean
binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Gaussian weighted
    cv2.THRESH_BINARY,
    11,  # Block size
    2    # Constant C
)
```

**Formula:**
```
threshold(x, y) = gaussian_weighted_mean(neighborhood) - C
```

### D. Parameter Tuning

#### Block Size

```python
# Small block size (7-11): For detailed images
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 7, 2)

# Medium block size (11-21): General purpose (recommended)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Large block size (21-41): For large uniform regions
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 2)
```

#### Constant C

```python
# Small C (0-2): Foreground expands
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 1)

# Medium C (2-5): Balanced (recommended)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 3)

# Large C (5-10): Background expands
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 7)
```

### E. Complete Example

```python
class AdaptiveThresholder:
    """Adaptive thresholding with auto-tuning"""

    def __init__(self, method='gaussian', block_size=11, C=2):
        self.method = method
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.C = C

    def threshold(self, image):
        """Apply adaptive threshold"""

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Choose method
        if self.method == 'mean':
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        else:  # gaussian
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

        # Apply threshold
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            adaptive_method,
            cv2.THRESH_BINARY,
            self.block_size,
            self.C
        )

        return binary

    def auto_tune(self, image):
        """Auto-tune parameters based on image characteristics"""

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Analyze image
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Adjust block size based on image size
        h, w = gray.shape
        image_size = h * w

        if image_size < 500 * 500:
            self.block_size = 7
        elif image_size < 1000 * 1000:
            self.block_size = 11
        else:
            self.block_size = 21

        # Adjust C based on std
        if std_brightness < 30:
            self.C = 2
        elif std_brightness < 60:
            self.C = 4
        else:
            self.C = 6

        print(f"Auto-tuned: block_size={self.block_size}, C={self.C}")

        return self.threshold(image)


# Usage
thresholder = AdaptiveThresholder(method='gaussian', block_size=11, C=2)

# Manual
binary = thresholder.threshold(image)

# Auto-tune
binary = thresholder.auto_tune(image)
```

---

## 5. Multi-Otsu Thresholding

### A. Multiple Thresholds

```python
from skimage.filters import threshold_multiotsu

# Calculate multiple thresholds
thresholds = threshold_multiotsu(gray, classes=3)  # 3 classes = 2 thresholds

print(f"Thresholds: {thresholds}")

# Apply thresholds
regions = np.digitize(gray, bins=thresholds)

# Visualize regions
result = np.zeros_like(gray)
for i in range(len(thresholds) + 1):
    result[regions == i] = int(255 * i / len(thresholds))
```

### B. Using OpenCV

```python
def multi_threshold_opencv(gray, n_classes=3):
    """Multi-level thresholding using OpenCV"""

    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Find peaks (simplified Otsu for multiple classes)
    # This is a simplified version; skimage is better for multi-Otsu

    # For demonstration: divide range into n_classes
    thresholds = []
    for i in range(1, n_classes):
        t = int(256 * i / n_classes)
        thresholds.append(t)

    # Apply thresholds
    result = np.zeros_like(gray)

    for i, t in enumerate(thresholds):
        if i == 0:
            result[gray <= t] = int(255 * (i + 1) / n_classes)
        else:
            result[(gray > thresholds[i-1]) & (gray <= t)] = int(255 * (i + 1) / n_classes)

    result[gray > thresholds[-1]] = 255

    return result, thresholds
```

---

## 6. Triangle Thresholding

### A. Method

Triangle method tốt cho images với **one dominant peak** và **long tail** trong histogram.

```python
from skimage.filters import threshold_triangle

# Calculate threshold
threshold_value = threshold_triangle(gray)

# Apply threshold
_, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
```

---

## 7. Color-Based Thresholding

### A. HSV Thresholding

```python
def threshold_by_color(image, lower_hsv, upper_hsv):
    """Threshold image by color range"""

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# Example: Detect red objects
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = threshold_by_color(image, lower_red1, upper_red1)
mask2 = threshold_by_color(image, lower_red2, upper_red2)

# Combine masks (red wraps around in HSV)
red_mask = cv2.bitwise_or(mask1, mask2)
```

### B. Interactive Color Picker

```python
def create_color_picker():
    """Interactive HSV color picker"""

    def nothing(x):
        pass

    cv2.namedWindow('Color Picker')

    # Create trackbars for HSV ranges
    cv2.createTrackbar('H_min', 'Color Picker', 0, 180, nothing)
    cv2.createTrackbar('H_max', 'Color Picker', 180, 180, nothing)
    cv2.createTrackbar('S_min', 'Color Picker', 0, 255, nothing)
    cv2.createTrackbar('S_max', 'Color Picker', 255, 255, nothing)
    cv2.createTrackbar('V_min', 'Color Picker', 0, 255, nothing)
    cv2.createTrackbar('V_max', 'Color Picker', 255, 255, nothing)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get trackbar values
        h_min = cv2.getTrackbarPos('H_min', 'Color Picker')
        h_max = cv2.getTrackbarPos('H_max', 'Color Picker')
        s_min = cv2.getTrackbarPos('S_min', 'Color Picker')
        s_max = cv2.getTrackbarPos('S_max', 'Color Picker')
        v_min = cv2.getTrackbarPos('V_min', 'Color Picker')
        v_max = cv2.getTrackbarPos('V_max', 'Color Picker')

        # Threshold
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = threshold_by_color(frame, lower, upper)

        # Display
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

---

## 8. CLAHE (Contrast Limited Adaptive Histogram Equalization)

### A. For Better Thresholding

```python
def threshold_with_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Threshold after CLAHE enhancement"""

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)

    # Otsu threshold on enhanced image
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary, enhanced
```

---

## 9. Comparison of Methods

| Method | Speed | Quality | Adaptive | Use Case |
|--------|-------|---------|----------|----------|
| Global | Fast ✅ | Poor | No | Uniform lighting |
| Otsu | Fast ✅ | Good | No | Bimodal histogram |
| Adaptive Mean | Medium | Good | Yes ✅ | Non-uniform light |
| Adaptive Gaussian | Medium | Better ✅ | Yes ✅ | Non-uniform light |
| Multi-Otsu | Medium | Good | No | Multiple classes |
| Triangle | Fast | Good | No | Skewed histogram |
| Color (HSV) | Fast | Excellent ✅ | No | Color segmentation |

---

## 10. Best Practices

### A. Pre-processing Pipeline

```python
def robust_threshold(image):
    """Robust thresholding pipeline"""

    # 1. Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 3. CLAHE (if low contrast)
    std = np.std(denoised)
    if std < 40:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        denoised = clahe.apply(denoised)

    # 4. Blur
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    # 5. Threshold
    #binary, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Alternative: Adaptive for non-uniform lighting
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 6. Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary
```

### B. Validation

```python
def evaluate_threshold(binary, ground_truth):
    """Evaluate thresholding result"""

    # Calculate metrics
    TP = np.sum((binary == 255) & (ground_truth == 255))
    FP = np.sum((binary == 255) & (ground_truth == 0))
    FN = np.sum((binary == 0) & (ground_truth == 255))
    TN = np.sum((binary == 0) & (ground_truth == 0))

    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

---

**Ngày tạo**: Tháng 1/2025
