# Image Processing Cơ Bản

## 1. Digital Image

### A. Định Nghĩa

Digital image là hàm 2D: `f(x, y)`

Trong đó:
- `(x, y)`: Tọa độ không gian
- `f`: Cường độ (intensity) hoặc màu sắc tại điểm đó

### B. Biểu Diễn

```python
import numpy as np

# Grayscale image (H × W)
gray_image = np.array([
    [0, 50, 100],
    [150, 200, 255]
])  # Shape: (2, 3)

# Color image (H × W × 3)
color_image = np.array([
    [[255, 0, 0], [0, 255, 0]],    # Row 1: Red, Green
    [[0, 0, 255], [255, 255, 0]]   # Row 2: Blue, Yellow
])  # Shape: (2, 2, 3) - BGR format
```

### C. Pixel Values

**Grayscale:**
```
Range: 0 to 255 (8-bit)
0   = Black (đen)
127 = Gray (xám)
255 = White (trắng)
```

**Color (BGR):**
```python
[255, 0, 0]     # Blue (xanh dương)
[0, 255, 0]     # Green (xanh lá)
[0, 0, 255]     # Red (đỏ)
[255, 255, 255] # White (trắng)
[0, 0, 0]       # Black (đen)
```

---

## 2. Image Properties

### A. Resolution

```python
# Load image
image = cv2.imread('image.jpg')
height, width, channels = image.shape

print(f"Resolution: {width} × {height}")
print(f"Channels: {channels}")
print(f"Total pixels: {width * height}")

# Example output:
# Resolution: 1920 × 1080
# Channels: 3
# Total pixels: 2,073,600
```

### B. Color Spaces

#### 1. BGR (OpenCV default)

```python
bgr_image = cv2.imread('image.jpg')
# Channels: Blue, Green, Red

b, g, r = cv2.split(bgr_image)
```

#### 2. RGB (Standard)

```python
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
# Channels: Red, Green, Blue
```

#### 3. Grayscale

```python
gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
# Single channel: Intensity only
```

#### 4. HSV (Hue, Saturation, Value)

```python
hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

# H: Hue (màu sắc) - 0 to 180
# S: Saturation (độ bão hòa) - 0 to 255
# V: Value (độ sáng) - 0 to 255
```

**Use Cases:**
- BGR/RGB: Display, general processing
- Grayscale: Edge detection, motion detection
- HSV: Color-based segmentation

---

## 3. Basic Operations

### A. Reading and Writing

```python
import cv2

# Read image
image = cv2.imread('input.jpg')

# Check if loaded
if image is None:
    print("Failed to load image")
else:
    print(f"Loaded: {image.shape}")

# Write image
cv2.imwrite('output.jpg', image)

# Write with quality (JPEG)
cv2.imwrite('output.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
```

### B. Display

```python
# Display image
cv2.imshow('Window Name', image)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()

# Display for 3 seconds
cv2.imshow('Window', image)
cv2.waitKey(3000)  # 3000 ms = 3 seconds
cv2.destroyAllWindows()
```

### C. Resizing

```python
# Resize to specific size
resized = cv2.resize(image, (640, 480))

# Resize by scale factor
scale = 0.5
width = int(image.shape[1] * scale)
height = int(image.shape[0] * scale)
resized = cv2.resize(image, (width, height))

# Resize with interpolation
resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
```

**Interpolation Methods:**
- `INTER_NEAREST`: Fastest, lowest quality
- `INTER_LINEAR`: Good balance (default)
- `INTER_CUBIC`: Slower, higher quality
- `INTER_AREA`: Best for downsampling

### D. Cropping

```python
# Crop region [y1:y2, x1:x2]
roi = image[100:300, 200:400]

# Crop center 50%
h, w = image.shape[:2]
center_x, center_y = w // 2, h // 2
crop_w, crop_h = w // 4, h // 4

cropped = image[
    center_y - crop_h : center_y + crop_h,
    center_x - crop_w : center_x + crop_w
]
```

---

## 4. Pixel Manipulation

### A. Accessing Pixels

```python
# Get pixel value (y, x)
pixel = image[100, 200]  # BGR values

# Grayscale
gray_pixel = gray_image[100, 200]  # Single value

# Set pixel value
image[100, 200] = [255, 0, 0]  # Set to blue
```

### B. Brightness Adjustment

```python
# Increase brightness
bright = cv2.add(image, 50)

# Decrease brightness
dark = cv2.subtract(image, 50)

# Using NumPy (with clipping)
bright = np.clip(image + 50, 0, 255).astype(np.uint8)
dark = np.clip(image - 50, 0, 255).astype(np.uint8)
```

### C. Contrast Adjustment

```python
# Increase contrast
contrast = cv2.multiply(image, 1.5)

# Formula: output = alpha * input + beta
alpha = 1.5  # Contrast (1.0-3.0)
beta = 0     # Brightness (-100 to 100)
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
```

---

## 5. Image Filtering

### A. Blurring

#### 1. Average Blur

```python
# Simple averaging
blurred = cv2.blur(image, (5, 5))
```

#### 2. Gaussian Blur

```python
# Weighted averaging (preferred)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Kernel size must be odd: 3, 5, 7, 9, ...
```

#### 3. Median Blur

```python
# Good for salt-and-pepper noise
denoised = cv2.medianBlur(image, 5)
```

#### 4. Bilateral Blur

```python
# Preserve edges while smoothing
smoothed = cv2.bilateralFilter(image, 9, 75, 75)
```

**Use Cases:**
```
Average: General smoothing
Gaussian: Remove Gaussian noise
Median: Remove salt-and-pepper noise
Bilateral: Smooth while keeping edges
```

### B. Sharpening

```python
# Sharpening kernel
kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

sharpened = cv2.filter2D(image, -1, kernel)
```

---

## 6. Morphological Operations

### A. Basic Operations

```python
# Define kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Erosion (làm mỏng)
eroded = cv2.erode(binary_image, kernel, iterations=1)

# Dilation (làm dày)
dilated = cv2.dilate(binary_image, kernel, iterations=1)

# Opening (erosion → dilation)
opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Closing (dilation → erosion)
closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
```

**Use Cases:**
```
Erosion:  Remove small white noise
Dilation: Fill small holes
Opening:  Remove small objects
Closing:  Fill small holes in objects
```

### B. Kernel Shapes

```python
# Rectangle
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Ellipse
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Cross
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
```

---

## 7. Thresholding

### A. Simple Threshold

```python
# Binary threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Thresholds:
# - THRESH_BINARY: > threshold → 255, else → 0
# - THRESH_BINARY_INV: > threshold → 0, else → 255
# - THRESH_TRUNC: > threshold → threshold, else → same
# - THRESH_TOZERO: > threshold → same, else → 0
```

### B. Otsu's Threshold

```python
# Automatic threshold selection
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### C. Adaptive Threshold

```python
# Local threshold
adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,  # Block size
    2    # Constant C
)
```

---

## 8. Histograms

### A. Calculate Histogram

```python
# Grayscale histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Color histogram
hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])  # Blue
hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])  # Green
hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])  # Red
```

### B. Histogram Equalization

```python
# Improve contrast
equalized = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
```

### C. Visualization

```python
import matplotlib.pyplot as plt

# Plot histogram
plt.hist(gray.ravel(), 256, [0, 256])
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
```

---

## 9. Drawing Functions

### A. Shapes

```python
# Line
cv2.line(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Rectangle
cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Circle
cv2.circle(image, (center_x, center_y), radius=30, color=(0, 0, 255), thickness=-1)  # -1 = filled

# Ellipse
cv2.ellipse(image, (center_x, center_y), (axis1, axis2), angle=0, startAngle=0, endAngle=360, color=(255, 0, 0), thickness=2)

# Polygon
pts = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32)
cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
```

### B. Text

```python
cv2.putText(
    image,
    'Hello World',
    (50, 50),  # Position
    cv2.FONT_HERSHEY_SIMPLEX,
    1,  # Font scale
    (255, 255, 255),  # Color
    2,  # Thickness
    cv2.LINE_AA  # Anti-aliased
)
```

---

## 10. Practical Examples

### Example 1: Image Enhancement

```python
def enhance_image(image):
    """Enhance image quality"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # Sharpen
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened
```

### Example 2: Simple Segmentation

```python
def segment_objects(image):
    """Segment objects from background"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned
```

### Example 3: Batch Processing

```python
import glob

def process_folder(input_folder, output_folder):
    """Process all images in folder"""
    # Get all images
    image_paths = glob.glob(f"{input_folder}/*.jpg")

    for img_path in image_paths:
        # Read
        image = cv2.imread(img_path)

        # Process
        processed = enhance_image(image)

        # Save
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed)

        print(f"Processed: {filename}")
```

---

## 11. Best Practices

### A. Memory Management

```python
# Release resources
cv2.destroyAllWindows()

# Delete large arrays
del large_image

# Force garbage collection
import gc
gc.collect()
```

### B. Type Safety

```python
# Always convert to uint8 for display/save
result = np.clip(processed, 0, 255).astype(np.uint8)

# Check data type
print(image.dtype)  # Should be uint8
```

### C. Error Handling

```python
def safe_imread(path):
    """Safely read image"""
    try:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load: {path}")
        return image
    except Exception as e:
        print(f"Error reading image: {e}")
        return None
```

---

## 12. Common Mistakes

### ❌ Wrong Color Space

```python
# Wrong: Display BGR in matplotlib
plt.imshow(cv2.imread('image.jpg'))  # Colors wrong!

# Correct: Convert to RGB
image = cv2.imread('image.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

### ❌ Array Out of Bounds

```python
# Wrong: No boundary check
pixel = image[1000, 1000]  # May crash!

# Correct: Check bounds
h, w = image.shape[:2]
if 0 <= y < h and 0 <= x < w:
    pixel = image[y, x]
```

### ❌ Incorrect Dimensions

```python
# Wrong: (x, y) vs (y, x) confusion
cv2.rectangle(image, (y1, x1), (y2, x2), color)  # Wrong!

# Correct: OpenCV uses (x, y)
cv2.rectangle(image, (x1, y1), (x2, y2), color)  # Correct
```

---

**Ngày tạo**: Tháng 1/2025
