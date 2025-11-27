# Image Processing (Xử Lý Ảnh) Cơ Bản

## 1. Digital Image (Ảnh Số)

### A. Định Nghĩa

Digital Image (ảnh số) là hàm 2D: `f(x, y)`

Trong đó:
- `(x, y)`: Tọa độ không gian
- `f`: Intensity (cường độ sáng) hoặc màu sắc tại điểm đó

### B. Biểu Diễn

```python
import numpy as np

# Grayscale image (ảnh xám) - H × W
gray_image = np.array([
    [0, 50, 100],
    [150, 200, 255]
])  # Shape: (2, 3)

# Color image (ảnh màu) - H × W × 3
color_image = np.array([
    [[255, 0, 0], [0, 255, 0]],    # Hàng 1: Đỏ, Xanh lá
    [[0, 0, 255], [255, 255, 0]]   # Hàng 2: Xanh dương, Vàng
])  # Shape: (2, 2, 3) - định dạng BGR
```

### C. Giá Trị Pixel

**Grayscale (ảnh xám):**
```
Khoảng: 0 đến 255 (8-bit)
0   = Đen
127 = Xám
255 = Trắng
```

**Color (ảnh màu - BGR):**
```python
[255, 0, 0]     # Xanh dương (Blue)
[0, 255, 0]     # Xanh lá (Green)
[0, 0, 255]     # Đỏ (Red)
[255, 255, 255] # Trắng
[0, 0, 0]       # Đen
```

---

## 2. Thuộc Tính Ảnh

### A. Resolution (Độ Phân Giải)

```python
# Đọc ảnh
image = cv2.imread('image.jpg')
height, width, channels = image.shape

print(f"Độ phân giải: {width} × {height}")
print(f"Kênh màu: {channels}")
print(f"Tổng số pixels: {width * height}")

# Ví dụ kết quả:
# Độ phân giải: 1920 × 1080
# Kênh màu: 3
# Tổng số pixels: 2,073,600
```

### B. Color Spaces (Không Gian Màu)

#### 1. BGR (Mặc định của OpenCV)

```python
bgr_image = cv2.imread('image.jpg')
# Các kênh: Blue (xanh dương), Green (xanh lá), Red (đỏ)

b, g, r = cv2.split(bgr_image)
```

#### 2. RGB (Chuẩn phổ biến)

```python
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
# Các kênh: Red, Green, Blue
```

#### 3. Grayscale (Ảnh xám)

```python
gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
# Chỉ 1 kênh: intensity
```

#### 4. HSV (Hue, Saturation, Value)

```python
hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

# H: Hue (màu sắc) - 0 đến 180
# S: Saturation (độ bão hòa) - 0 đến 255
# V: Value (độ sáng) - 0 đến 255
```

**Trường hợp sử dụng:**
- BGR/RGB: Hiển thị, xử lý thông thường
- Grayscale: Phát hiện cạnh, phát hiện chuyển động
- HSV: Phân vùng dựa trên màu sắc

---

## 3. Các Thao Tác Cơ Bản

### A. Đọc và Ghi File

```python
import cv2

# Đọc ảnh
image = cv2.imread('input.jpg')

# Kiểm tra đã tải thành công chưa
if image is None:
    print("Không thể tải ảnh")
else:
    print(f"Đã tải: {image.shape}")

# Ghi ảnh
cv2.imwrite('output.jpg', image)

# Ghi với chất lượng JPEG tùy chỉnh
cv2.imwrite('output.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
```

### B. Hiển Thị

```python
# Hiển thị ảnh
cv2.imshow('Tên cửa sổ', image)
cv2.waitKey(0)  # Đợi nhấn phím
cv2.destroyAllWindows()

# Hiển thị trong 3 giây
cv2.imshow('Cửa sổ', image)
cv2.waitKey(3000)  # 3000 ms = 3 giây
cv2.destroyAllWindows()
```

### C. Thay Đổi Kích Thước

```python
# Resize về kích thước cụ thể
resized = cv2.resize(image, (640, 480))

# Resize theo tỷ lệ
scale = 0.5
width = int(image.shape[1] * scale)
height = int(image.shape[0] * scale)
resized = cv2.resize(image, (width, height))

# Resize với interpolation
resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
```

**Phương pháp Interpolation (Nội suy):**
- `INTER_NEAREST`: Nhanh nhất, chất lượng thấp nhất
- `INTER_LINEAR`: Cân bằng tốt (mặc định)
- `INTER_CUBIC`: Chậm hơn, chất lượng cao hơn
- `INTER_AREA`: Tốt nhất cho downsampling (giảm kích thước)

### D. Cắt Ảnh

```python
# Cắt vùng [y1:y2, x1:x2]
roi = image[100:300, 200:400]

# Cắt 50% ở giữa
h, w = image.shape[:2]
center_x, center_y = w // 2, h // 2
crop_w, crop_h = w // 4, h // 4

cropped = image[
    center_y - crop_h : center_y + crop_h,
    center_x - crop_w : center_x + crop_w
]
```

---

## 4. Thao Tác Với Pixel

### A. Truy Cập Pixel

```python
# Lấy giá trị pixel (y, x)
pixel = image[100, 200]  # Giá trị BGR

# Ảnh xám
gray_pixel = gray_image[100, 200]  # Giá trị đơn

# Đặt giá trị pixel
image[100, 200] = [255, 0, 0]  # Đặt thành màu xanh dương
```

### B. Điều Chỉnh Độ Sáng

```python
# Tăng độ sáng
bright = cv2.add(image, 50)

# Giảm độ sáng
dark = cv2.subtract(image, 50)

# Sử dụng NumPy (với clipping)
bright = np.clip(image + 50, 0, 255).astype(np.uint8)
dark = np.clip(image - 50, 0, 255).astype(np.uint8)
```

### C. Điều Chỉnh Độ Tương Phản

```python
# Tăng độ tương phản
contrast = cv2.multiply(image, 1.5)

# Công thức: output = alpha * input + beta
alpha = 1.5  # Độ tương phản (1.0-3.0)
beta = 0     # Độ sáng (-100 đến 100)
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
```

---

## 5. Image Filtering (Lọc Ảnh)

### A. Làm Mờ (Blurring)

#### 1. Average Blur (Làm mờ trung bình)

```python
# Trung bình đơn giản
blurred = cv2.blur(image, (5, 5))
```

#### 2. Gaussian Blur (Làm mờ Gaussian)

```python
# Trung bình có trọng số (được ưu tiên)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Kích thước kernel phải lẻ: 3, 5, 7, 9, ...
```

#### 3. Median Blur (Làm mờ trung vị)

```python
# Tốt cho nhiễu salt-and-pepper
denoised = cv2.medianBlur(image, 5)
```

#### 4. Bilateral Filter (Lọc song phương)

```python
# Giữ cạnh trong khi làm mịn
smoothed = cv2.bilateralFilter(image, 9, 75, 75)
```

**Trường hợp sử dụng:**
```
Average: Làm mịn chung
Gaussian: Loại bỏ nhiễu Gaussian
Median: Loại bỏ nhiễu salt-and-pepper
Bilateral: Làm mịn nhưng giữ cạnh
```

### B. Làm Sắc Nét (Sharpening)

```python
# Kernel làm sắc nét
kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

sharpened = cv2.filter2D(image, -1, kernel)
```

---

## 6. Morphological Operations (Các Phép Toán Hình Thái)

### A. Các Phép Toán Cơ Bản

```python
# Định nghĩa kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Erosion (ăn mòn - làm mỏng)
eroded = cv2.erode(binary_image, kernel, iterations=1)

# Dilation (giãn nở - làm dày)
dilated = cv2.dilate(binary_image, kernel, iterations=1)

# Opening (erosion → dilation)
opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Closing (dilation → erosion)
closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
```

**Trường hợp sử dụng:**
```
Erosion:  Loại bỏ nhiễu trắng nhỏ
Dilation: Lấp đầy lỗ nhỏ
Opening:  Loại bỏ vật thể nhỏ
Closing:  Lấp đầy lỗ nhỏ trong vật thể
```

### B. Hình Dạng Kernel

```python
# Hình chữ nhật
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Hình elip
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Hình chữ thập
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
```

---

## 7. Thresholding (Ngưỡng Hóa)

### A. Ngưỡng Đơn Giản

```python
# Ngưỡng nhị phân
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Các loại ngưỡng:
# - THRESH_BINARY: > ngưỡng → 255, ngược lại → 0
# - THRESH_BINARY_INV: > ngưỡng → 0, ngược lại → 255
# - THRESH_TRUNC: > ngưỡng → ngưỡng, ngược lại → giữ nguyên
# - THRESH_TOZERO: > ngưỡng → giữ nguyên, ngược lại → 0
```

### B. Otsu's Threshold (Ngưỡng Otsu)

```python
# Tự động chọn ngưỡng
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### C. Adaptive Threshold (Ngưỡng Thích Ứng)

```python
# Ngưỡng cục bộ
adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,  # Kích thước block
    2    # Hằng số C
)
```

---

## 8. Histograms (Biểu Đồ)

### A. Tính Histogram

```python
# Histogram ảnh xám
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Histogram màu
hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])  # Blue
hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])  # Green
hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])  # Red
```

### B. Histogram Equalization (Cân Bằng Histogram)

```python
# Cải thiện độ tương phản
equalized = cv2.equalizeHist(gray)

# CLAHE (Contrast Limited Adaptive Histogram Equalization - Cân bằng histogram thích ứng có giới hạn tương phản)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
```

### C. Trực Quan Hóa

```python
import matplotlib.pyplot as plt

# Vẽ histogram
plt.hist(gray.ravel(), 256, [0, 256])
plt.title('Histogram Ảnh Xám')
plt.xlabel('Giá Trị Pixel')
plt.ylabel('Tần Suất')
plt.show()
```

---

## 9. Các Hàm Vẽ

### A. Hình Dạng

```python
# Đường thẳng
cv2.line(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Hình chữ nhật
cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

# Hình tròn
cv2.circle(image, (center_x, center_y), radius=30, color=(0, 0, 255), thickness=-1)  # -1 = tô đầy

# Hình elip
cv2.ellipse(image, (center_x, center_y), (axis1, axis2), angle=0, startAngle=0, endAngle=360, color=(255, 0, 0), thickness=2)

# Đa giác
pts = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], np.int32)
cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
```

### B. Văn Bản

```python
cv2.putText(
    image,
    'Hello World',
    (50, 50),  # Vị trí
    cv2.FONT_HERSHEY_SIMPLEX,
    1,  # Kích thước font
    (255, 255, 255),  # Màu
    2,  # Độ dày
    cv2.LINE_AA  # Anti-aliased (khử răng cưa)
)
```

---

## 10. Ví Dụ Thực Tế

### Ví dụ 1: Cải Thiện Chất Lượng Ảnh

```python
def enhance_image(image):
    """Cải thiện chất lượng ảnh"""
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Khử nhiễu
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

    # Làm sắc nét
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened
```

### Ví dụ 2: Phân Vùng Đơn Giản

```python
def segment_objects(image):
    """Phân vùng vật thể khỏi nền"""
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ngưỡng hóa
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Phép toán hình thái
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned
```

### Ví dụ 3: Xử Lý Hàng Loạt

```python
import glob

def process_folder(input_folder, output_folder):
    """Xử lý tất cả ảnh trong thư mục"""
    # Lấy tất cả ảnh
    image_paths = glob.glob(f"{input_folder}/*.jpg")

    for img_path in image_paths:
        # Đọc
        image = cv2.imread(img_path)

        # Xử lý
        processed = enhance_image(image)

        # Lưu
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed)

        print(f"Đã xử lý: {filename}")
```

---

## 11. Best Practices (Thực Hành Tốt)

### A. Quản Lý Bộ Nhớ

```python
# Giải phóng tài nguyên
cv2.destroyAllWindows()

# Xóa mảng lớn
del large_image

# Ép buộc garbage collection
import gc
gc.collect()
```

### B. An Toàn Kiểu Dữ Liệu

```python
# Luôn chuyển về uint8 để hiển thị/lưu
result = np.clip(processed, 0, 255).astype(np.uint8)

# Kiểm tra kiểu dữ liệu
print(image.dtype)  # Nên là uint8
```

### C. Xử Lý Lỗi

```python
def safe_imread(path):
    """Đọc ảnh an toàn"""
    try:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Không thể tải: {path}")
        return image
    except Exception as e:
        print(f"Lỗi đọc ảnh: {e}")
        return None
```

---

## 12. Lỗi Thường Gặp

### ❌ Sai Color Space (Không Gian Màu)

```python
# Sai: Hiển thị BGR trong matplotlib
plt.imshow(cv2.imread('image.jpg'))  # Màu sai!

# Đúng: Chuyển sang RGB
image = cv2.imread('image.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

### ❌ Vượt Quá Giới Hạn Mảng

```python
# Sai: Không kiểm tra biên
pixel = image[1000, 1000]  # Có thể bị crash!

# Đúng: Kiểm tra biên
h, w = image.shape[:2]
if 0 <= y < h and 0 <= x < w:
    pixel = image[y, x]
```

### ❌ Nhầm Lẫn Chiều

```python
# Sai: Nhầm lẫn (x, y) vs (y, x)
cv2.rectangle(image, (y1, x1), (y2, x2), color)  # Sai!

# Đúng: OpenCV dùng (x, y)
cv2.rectangle(image, (x1, y1), (x2, y2), color)  # Đúng
```

---

**Ngày tạo**: Tháng 1/2025
