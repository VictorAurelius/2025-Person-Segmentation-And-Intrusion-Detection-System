# Thresholding Techniques (Kỹ Thuật Ngưỡng Hóa)

## 1. Khái Niệm

**Thresholding (ngưỡng hóa)** là kỹ thuật phân vùng đơn giản nhất, phân chia ảnh thành nền trước và nền sau dựa trên cường độ sáng pixel.

### A. Nguyên Tắc Cơ Bản

```python
if pixel_value > threshold:
    output = 255  # Nền trước (trắng)
else:
    output = 0    # Nền sau (đen)
```

---

## 2. Global Thresholding (Ngưỡng Hóa Toàn Cục)

### A. Binary Threshold (Ngưỡng Nhị Phân)

```python
import cv2
import numpy as np

# Tải ảnh
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Áp dụng ngưỡng nhị phân
threshold_value = 127
_, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

cv2.imshow('Gốc', image)
cv2.imshow('Nhị phân', binary)
cv2.waitKey(0)
```

**Các Loại Ngưỡng:**

```python
# THRESH_BINARY: > ngưỡng → 255, ngược lại → 0
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# THRESH_BINARY_INV: > ngưỡng → 0, ngược lại → 255
_, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# THRESH_TRUNC: > ngưỡng → ngưỡng, ngược lại → giữ nguyên
_, trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

# THRESH_TOZERO: > ngưỡng → giữ nguyên, ngược lại → 0
_, tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

# THRESH_TOZERO_INV: > ngưỡng → 0, ngược lại → giữ nguyên
_, tozero_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
```

### B. Với Trackbar (Tương Tác)

```python
def nothing(x):
    pass

cv2.namedWindow('Ngưỡng')
cv2.createTrackbar('Ngưỡng', 'Ngưỡng', 127, 255, nothing)

while True:
    # Lấy vị trí trackbar
    threshold_value = cv2.getTrackbarPos('Ngưỡng', 'Ngưỡng')

    # Áp dụng ngưỡng
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imshow('Ngưỡng', binary)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

---

## 3. Otsu's Thresholding (Ngưỡng Otsu)

### A. Tự Động Chọn Ngưỡng

**Phương pháp Otsu** tự động tìm ngưỡng tối ưu bằng cách tối đa hóa phương sai giữa các lớp.

```python
# Ngưỡng Otsu
threshold_value, binary = cv2.threshold(
    gray,
    0,        # Giá trị ngưỡng (bỏ qua, tự động tính)
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"Ngưỡng Otsu: {threshold_value}")
```

### B. Với Gaussian Blur (Làm Mờ Gaussian)

```python
# Làm mờ trước Otsu (khuyến nghị)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
threshold_value, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

### C. Ví Dụ Hoàn Chỉnh

```python
def otsu_segmentation(image):
    """Phân vùng ảnh sử dụng phương pháp Otsu"""

    # Chuyển sang ảnh xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Khử nhiễu
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Làm mờ
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    # Ngưỡng Otsu
    threshold_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Hình thái (tùy chọn)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary, threshold_value
```

---

## 4. Adaptive Thresholding (Ngưỡng Thích Ứng)

### A. Động Lực

**Vấn Đề Với Ngưỡng Toàn Cục:**
- Ánh sáng không đồng đều
- Bóng đổ
- Độ chuyển màu

**Giải Pháp:** Ngưỡng cục bộ khác nhau cho từng vùng

### B. Adaptive Mean (Trung Bình Thích Ứng)

```python
# Ngưỡng thích ứng với trung bình
binary = cv2.adaptiveThreshold(
    gray,
    255,                        # Giá trị max
    cv2.ADAPTIVE_THRESH_MEAN_C, # Phương pháp
    cv2.THRESH_BINARY,          # Loại
    11,                         # Kích thước khối (phải lẻ)
    2                           # Hằng số C
)
```

**Công Thức:**
```
ngưỡng(x, y) = trung_bình(vùng_lân_cận) - C
```

### C. Adaptive Gaussian (Gaussian Thích Ứng)

```python
# Ngưỡng thích ứng với trung bình có trọng số Gaussian
binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Trọng số Gaussian
    cv2.THRESH_BINARY,
    11,  # Kích thước khối
    2    # Hằng số C
)
```

**Công Thức:**
```
ngưỡng(x, y) = trung_bình_trọng_số_gaussian(vùng_lân_cận) - C
```

### D. Điều Chỉnh Tham Số

#### Kích Thước Khối

```python
# Khối nhỏ (7-11): Cho ảnh chi tiết
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 7, 2)

# Khối trung bình (11-21): Mục đích chung (khuyến nghị)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Khối lớn (21-41): Cho vùng đồng nhất lớn
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 2)
```

#### Hằng Số C

```python
# C nhỏ (0-2): Nền trước mở rộng
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 1)

# C trung bình (2-5): Cân bằng (khuyến nghị)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 3)

# C lớn (5-10): Nền sau mở rộng
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 7)
```

### E. Ví Dụ Hoàn Chỉnh

```python
class AdaptiveThresholder:
    """Ngưỡng thích ứng với tự động điều chỉnh"""

    def __init__(self, method='gaussian', block_size=11, C=2):
        self.method = method
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.C = C

    def threshold(self, image):
        """Áp dụng ngưỡng thích ứng"""

        # Chuyển sang ảnh xám
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Chọn phương pháp
        if self.method == 'mean':
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        else:  # gaussian
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

        # Áp dụng ngưỡng
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
        """Tự động điều chỉnh tham số dựa trên đặc tính ảnh"""

        # Chuyển sang ảnh xám
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Phân tích ảnh
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Điều chỉnh kích thước khối dựa trên kích thước ảnh
        h, w = gray.shape
        image_size = h * w

        if image_size < 500 * 500:
            self.block_size = 7
        elif image_size < 1000 * 1000:
            self.block_size = 11
        else:
            self.block_size = 21

        # Điều chỉnh C dựa trên độ lệch chuẩn
        if std_brightness < 30:
            self.C = 2
        elif std_brightness < 60:
            self.C = 4
        else:
            self.C = 6

        print(f"Tự động điều chỉnh: block_size={self.block_size}, C={self.C}")

        return self.threshold(image)


# Sử dụng
thresholder = AdaptiveThresholder(method='gaussian', block_size=11, C=2)

# Thủ công
binary = thresholder.threshold(image)

# Tự động điều chỉnh
binary = thresholder.auto_tune(image)
```

---

## 5. Multi-Otsu Thresholding (Ngưỡng Đa Otsu)

### A. Ngưỡng Đa Mức

```python
from skimage.filters import threshold_multiotsu

# Tính nhiều ngưỡng
thresholds = threshold_multiotsu(gray, classes=3)  # 3 lớp = 2 ngưỡng

print(f"Ngưỡng: {thresholds}")

# Áp dụng ngưỡng
regions = np.digitize(gray, bins=thresholds)

# Trực quan hóa vùng
result = np.zeros_like(gray)
for i in range(len(thresholds) + 1):
    result[regions == i] = int(255 * i / len(thresholds))
```

### B. Sử Dụng OpenCV

```python
def multi_threshold_opencv(gray, n_classes=3):
    """Ngưỡng đa mức sử dụng OpenCV"""

    # Tính histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Tìm đỉnh (Otsu đơn giản hóa cho nhiều lớp)
    # Đây là phiên bản đơn giản; skimage tốt hơn cho multi-Otsu

    # Để minh họa: chia phạm vi thành n_classes
    thresholds = []
    for i in range(1, n_classes):
        t = int(256 * i / n_classes)
        thresholds.append(t)

    # Áp dụng ngưỡng
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

## 6. Triangle Thresholding (Ngưỡng Tam Giác)

### A. Phương Pháp

Phương pháp tam giác tốt cho ảnh có **một đỉnh chủ đạo** và **đuôi dài** trong histogram.

```python
from skimage.filters import threshold_triangle

# Tính ngưỡng
threshold_value = threshold_triangle(gray)

# Áp dụng ngưỡng
_, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
```

---

## 7. Color-Based Thresholding (Ngưỡng Theo Màu)

### A. HSV Thresholding (Ngưỡng HSV)

```python
def threshold_by_color(image, lower_hsv, upper_hsv):
    """Ngưỡng ảnh theo phạm vi màu"""

    # Chuyển sang HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tạo mặt nạ
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Hình thái
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# Ví dụ: Phát hiện đối tượng màu đỏ
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = threshold_by_color(image, lower_red1, upper_red1)
mask2 = threshold_by_color(image, lower_red2, upper_red2)

# Kết hợp mặt nạ (đỏ quấn quanh trong HSV)
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

## 9. So Sánh Các Phương Pháp

| Phương Pháp | Tốc Độ | Chất Lượng | Thích Ứng | Trường Hợp Sử Dụng |
|--------|-------|---------|----------|----------|
| Toàn Cục | Nhanh ✅ | Kém | Không | Ánh sáng đồng đều |
| Otsu | Nhanh ✅ | Tốt | Không | Histogram hai đỉnh |
| Adaptive Mean | Trung Bình | Tốt | Có ✅ | Ánh sáng không đồng đều |
| Adaptive Gaussian | Trung Bình | Tốt Hơn ✅ | Có ✅ | Ánh sáng không đồng đều |
| Multi-Otsu | Trung Bình | Tốt | Không | Nhiều lớp |
| Triangle | Nhanh | Tốt | Không | Histogram lệch |
| Color (HSV) | Nhanh | Xuất Sắc ✅ | Không | Phân vùng màu |

---

## 10. Best Practices (Thực Hành Tốt)

### A. Pre-processing Pipeline (Quy Trình Tiền Xử Lý)

```python
def robust_threshold(image):
    """Quy trình ngưỡng hóa mạnh mẽ"""

    # 1. Chuyển sang ảnh xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 2. Khử nhiễu
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 3. CLAHE (nếu độ tương phản thấp)
    std = np.std(denoised)
    if std < 40:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        denoised = clahe.apply(denoised)

    # 4. Làm mờ
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    # 5. Ngưỡng hóa
    #binary, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Thay thế: Thích ứng cho ánh sáng không đồng đều
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 6. Hình thái
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary
```

### B. Validation (Xác Thực)

```python
def evaluate_threshold(binary, ground_truth):
    """Đánh giá kết quả ngưỡng hóa"""

    # Tính các chỉ số
    TP = np.sum((binary == 255) & (ground_truth == 255))
    FP = np.sum((binary == 255) & (ground_truth == 0))
    FN = np.sum((binary == 0) & (ground_truth == 255))
    TN = np.sum((binary == 0) & (ground_truth == 0))

    # Chỉ số
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
