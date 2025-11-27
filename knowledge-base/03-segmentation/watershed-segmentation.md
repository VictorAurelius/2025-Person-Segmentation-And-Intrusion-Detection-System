# Watershed Segmentation (Phân Vùng Watershed)

## 1. Khái Niệm

**Watershed Segmentation (phân vùng watershed/phân thủy)** là thuật toán dựa trên diễn giải topo của ảnh, xử lý ảnh xám như một bề mặt địa hình.

### A. Ẩn Dụ

Tưởng tượng ảnh xám như địa hình:
- **Vùng sáng** = Đỉnh núi
- **Vùng tối** = Thung lũng
- **Gradients (độ dốc)** = Sườn núi

Nếu đổ nước vào thung lũng, nước sẽ dâng lên và tạo thành các **watersheds (phân thủy)**.

---

## 2. Nguyên Lý Thuật Toán

### A. Quá Trình Ngập Lụt

```
1. Tìm cực tiểu cục bộ (thung lũng)
2. Đánh dấu mỗi thung lũng với nhãn duy nhất
3. "Đổ nước" vào thung lũng (ngập lụt)
4. Khi 2 "vùng nước" gặp nhau → Đường phân thủy
5. Watersheds tạo thành ranh giới giữa các đối tượng
```

### B. Trực Quan Hóa

```
Biểu đồ Cường độ Ảnh:
      ╱╲        ╱╲
     ╱  ╲      ╱  ╲
    ╱    ╲____╱    ╲
   ╱                ╲
  ╱                  ╲

→ Thung lũng (markers) tại ____
→ Nước dâng lên ↑
→ Watershed tại đỉnh ╱╲
```

---

## 3. Triển Khai Cơ Bản

### A. Watershed Đơn Giản

```python
import cv2
import numpy as np
from scipy import ndimage

# Đọc ảnh
image = cv2.imread('coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ngưỡng hóa
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Loại bỏ nhiễu
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Nền sau chắc chắn (giãn nở)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Nền trước chắc chắn (biến đổi khoảng cách)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Vùng không xác định
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Gắn nhãn markers
_, markers = cv2.connectedComponents(sure_fg)

# Thêm 1 vào tất cả nhãn (nền = 1, không phải 0)
markers = markers + 1

# Đánh dấu vùng không xác định là 0
markers[unknown == 255] = 0

# Áp dụng watershed
markers = cv2.watershed(image, markers)

# Đánh dấu ranh giới
image[markers == -1] = [255, 0, 0]  # Ranh giới màu đỏ

cv2.imshow('Kết quả', image)
cv2.waitKey(0)
```

---

## 4. Marker-Controlled Watershed (Watershed Điều Khiển Bằng Markers)

### A. Tại Sao Cần Markers?

**Vấn Đề:** Watershed cơ bản phân vùng quá mức (quá nhiều vùng)

**Giải Pháp:** Sử dụng markers để hướng dẫn phân vùng

### B. Triển Khai

```python
class WatershedSegmentation:
    """Phân vùng watershed điều khiển bằng markers"""

    def __init__(self):
        self.markers = None

    def segment(self, image, min_distance=20):
        """Phân vùng ảnh sử dụng watershed"""

        # Chuẩn bị ảnh
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ngưỡng hóa
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Loại bỏ nhiễu
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Nền sau chắc chắn
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Biến đổi khoảng cách
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        # Nền trước chắc chắn
        _, sure_fg = cv2.threshold(
            dist_transform,
            0.5 * dist_transform.max(),
            255,
            0
        )
        sure_fg = np.uint8(sure_fg)

        # Vùng không xác định
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Gắn nhãn markers
        _, self.markers = cv2.connectedComponents(sure_fg)

        # Thêm 1 vào markers (nền != 0)
        self.markers = self.markers + 1

        # Đánh dấu vùng không xác định là 0
        self.markers[unknown == 255] = 0

        # Áp dụng watershed
        self.markers = cv2.watershed(image, self.markers)

        return self.markers

    def visualize(self, image, markers=None):
        """Trực quan hóa kết quả phân vùng"""

        if markers is None:
            markers = self.markers

        if markers is None:
            raise ValueError("Không có markers. Chạy segment() trước.")

        result = image.copy()

        # Tô màu mỗi vùng
        num_segments = markers.max()

        for i in range(2, num_segments + 1):
            mask = (markers == i).astype(np.uint8)

            # Màu ngẫu nhiên
            color = np.random.randint(0, 255, 3).tolist()

            # Áp dụng màu
            result[mask == 1] = color

        # Đánh dấu ranh giới
        result[markers == -1] = [255, 0, 0]

        return result

    def get_contours(self, markers=None):
        """Trích xuất đường viền từ markers"""

        if markers is None:
            markers = self.markers

        contours = []
        num_segments = markers.max()

        for i in range(2, num_segments + 1):
            mask = (markers == i).astype(np.uint8) * 255

            # Tìm đường viền
            cnts, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(cnts) > 0:
                contours.append(cnts[0])

        return contours


# Sử dụng
segmenter = WatershedSegmentation()

# Phân vùng
markers = segmenter.segment(image)

# Trực quan hóa
result = segmenter.visualize(image)
cv2.imshow('Phân Vùng Watershed', result)

# Lấy đường viền
contours = segmenter.get_contours()
print(f"Tìm thấy {len(contours)} đối tượng")
```

---

## 5. Distance Transform (Biến Đổi Khoảng Cách)

### A. Khái Niệm

Distance Transform tính khoảng cách từ mỗi pixel nền trước đến pixel nền sau gần nhất.

```python
# Ảnh nhị phân
binary = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=np.uint8)

# Biến đổi khoảng cách
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Kết quả (xấp xỉ):
# [0, 0, 0, 0, 0]
# [0, 1, 1, 1, 0]
# [0, 1, 2, 1, 0]  ← Tâm có khoảng cách cao nhất
# [0, 1, 1, 1, 0]
# [0, 0, 0, 0, 0]
```

### B. Độ Đo Khoảng Cách

```python
# Euclidean (L2)
dist_l2 = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Manhattan (L1)
dist_l1 = cv2.distanceTransform(binary, cv2.DIST_L1, 3)

# Chessboard (L∞)
dist_linf = cv2.distanceTransform(binary, cv2.DIST_C, 3)
```

### C. Phát Hiện Đỉnh

```python
def find_peaks(dist_transform, min_distance=20):
    """Tìm đỉnh trong biến đổi khoảng cách"""

    from scipy.ndimage import maximum_filter

    # Cực đại cục bộ
    local_max = maximum_filter(dist_transform, size=min_distance) == dist_transform

    # Ngưỡng (chỉ đỉnh đáng kể)
    threshold = 0.5 * dist_transform.max()
    peaks = local_max & (dist_transform > threshold)

    # Lấy tọa độ
    coords = np.argwhere(peaks)

    return coords
```

---

## 6. Interactive Watershed

### A. Manual Marker Placement

```python
class InteractiveWatershed:
    """Interactive watershed with manual markers"""

    def __init__(self, image):
        self.image = image.copy()
        self.markers_image = np.zeros(image.shape[:2], dtype=np.int32)
        self.marker_id = 1
        self.drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            # Draw marker
            cv2.circle(self.markers_image, (x, y), 5, self.marker_id, -1)
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            cv2.circle(self.markers_image, (x, y), 5, self.marker_id, -1)
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def run(self):
        """Run interactive segmentation"""

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)

        original = self.image.copy()

        while True:
            cv2.imshow('Image', self.image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):  # New marker
                self.marker_id += 1
                print(f"Marker {self.marker_id}")

            elif key == ord('r'):  # Reset
                self.image = original.copy()
                self.markers_image = np.zeros(self.image.shape[:2], dtype=np.int32)
                self.marker_id = 1
                print("Reset")

            elif key == ord('w'):  # Run watershed
                # Apply watershed
                markers = cv2.watershed(original, self.markers_image.copy())

                # Visualize
                result = self.visualize(original, markers)
                cv2.imshow('Result', result)

            elif key == ord('q'):  # Quit
                break

        cv2.destroyAllWindows()

    def visualize(self, image, markers):
        """Visualize segmentation"""

        result = image.copy()

        # Color segments
        for i in range(1, self.marker_id + 1):
            mask = (markers == i).astype(np.uint8)
            color = np.random.randint(0, 255, 3).tolist()
            result[mask == 1] = color

        # Boundaries
        result[markers == -1] = [255, 0, 0]

        return result


# Usage
interactive = InteractiveWatershed(image)
interactive.run()
```

---

## 7. Gradient-Based Watershed

### A. Using Gradients

```python
def gradient_watershed(image):
    """Watershed on gradient image"""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate gradient
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    # Threshold gradient
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Find markers
    _, markers = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    markers = np.uint8(markers)

    # Label markers
    _, markers = cv2.connectedComponents(markers)

    # Apply watershed on gradient
    markers = cv2.watershed(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), markers)

    return markers
```

---

## 8. Tách Các Đối Tượng Chạm Nhau

### A. Vấn Đề

Các đối tượng chạm nhau/chồng lên nhau xuất hiện như một khối duy nhất.

### B. Giải Pháp

```python
def separate_touching_objects(binary_mask):
    """Tách đối tượng chạm nhau sử dụng watershed"""

    # Biến đổi khoảng cách
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Tìm đỉnh (tâm đối tượng)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Giãn nở để tìm nền
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)

    # Vùng không xác định
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Gắn nhãn markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Áp dụng watershed
    # Cần ảnh 3 kênh
    image_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_3ch, markers)

    # Tạo mặt nạ đã tách
    separated = np.zeros_like(binary_mask)
    separated[markers > 1] = 255

    return separated, markers
```

---

## 9. Điều Chỉnh Tham Số

### A. Ngưỡng Biến Đổi Khoảng Cách

```python
# Ngưỡng thấp (0.3-0.5): Tách mạnh hơn
_, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

# Ngưỡng trung bình (0.5-0.7): Cân bằng (khuyến nghị)
_, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

# Ngưỡng cao (0.7-0.9): Tách bảo thủ
_, sure_fg = cv2.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)
```

### B. Phép Toán Hình Thái

```python
# Opening yếu (loại bỏ nhiễu ít)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# Opening mạnh (loại bỏ nhiễu nhiều)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)

# Dilation nhỏ (nền chặt)
sure_bg = cv2.dilate(opening, kernel, iterations=1)

# Dilation lớn (nền rộng)
sure_bg = cv2.dilate(opening, kernel, iterations=5)
```

---

## 10. Advanced Techniques

### A. Multi-Scale Watershed

```python
def multiscale_watershed(image, scales=[1.0, 0.75, 0.5]):
    """Watershed at multiple scales"""

    all_markers = []

    for scale in scales:
        # Resize
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Segment
        segmenter = WatershedSegmentation()
        markers = segmenter.segment(resized)

        # Resize markers back
        markers = cv2.resize(markers.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        all_markers.append(markers)

    # Combine markers (voting or averaging)
    combined = np.mean(all_markers, axis=0).astype(np.int32)

    return combined
```

### B. Watershed with Color

```python
def color_watershed(image):
    """Watershed using color information"""

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Calculate gradient in LAB space
    l_grad = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_GRADIENT, np.ones((3, 3)))
    a_grad = cv2.morphologyEx(lab[:, :, 1], cv2.MORPH_GRADIENT, np.ones((3, 3)))
    b_grad = cv2.morphologyEx(lab[:, :, 2], cv2.MORPH_GRADIENT, np.ones((3, 3)))

    # Combine gradients
    gradient = np.maximum(np.maximum(l_grad, a_grad), b_grad)

    # Threshold
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Standard watershed process
    # ... (distance transform, markers, watershed)

    return markers
```

---

## 11. Hậu Xử Lý

### A. Gộp Vùng Nhỏ

```python
def merge_small_regions(markers, min_size=100):
    """Gộp các vùng nhỏ hơn min_size"""

    unique_markers = np.unique(markers)
    unique_markers = unique_markers[unique_markers > 1]  # Loại trừ 0, 1, -1

    for marker_id in unique_markers:
        region_mask = (markers == marker_id)
        region_size = np.sum(region_mask)

        if region_size < min_size:
            # Tìm vùng lân cận
            dilated = cv2.dilate(region_mask.astype(np.uint8), np.ones((3, 3)))
            neighbors = markers[dilated == 1]
            neighbors = neighbors[neighbors != marker_id]
            neighbors = neighbors[neighbors > 1]

            if len(neighbors) > 0:
                # Gộp với vùng lân cận phổ biến nhất
                merge_to = np.bincount(neighbors).argmax()
                markers[region_mask] = merge_to

    return markers
```

### B. Làm Mượt Ranh Giới

```python
def smooth_watershed_boundaries(markers):
    """Làm mượt ranh giới watershed"""

    smoothed = markers.copy()

    # Cho mỗi vùng
    unique_markers = np.unique(markers)
    unique_markers = unique_markers[unique_markers > 1]

    for marker_id in unique_markers:
        # Lấy mặt nạ vùng
        mask = (markers == marker_id).astype(np.uint8) * 255

        # Tìm đường viền
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Làm mượt đường viền
            epsilon = 0.01 * cv2.arcLength(contours[0], True)
            smoothed_contour = cv2.approxPolyDP(contours[0], epsilon, True)

            # Cập nhật markers
            mask_new = np.zeros_like(mask)
            cv2.drawContours(mask_new, [smoothed_contour], 0, 255, -1)

            smoothed[mask_new == 255] = marker_id

    return smoothed
```

---

## 12. Ví Dụ Hoàn Chỉnh

```python
class AdvancedWatershed:
    """Quy trình phân vùng watershed nâng cao"""

    def __init__(self, min_area=100, dist_threshold=0.6):
        self.min_area = min_area
        self.dist_threshold = dist_threshold

    def segment(self, image):
        """Quy trình phân vùng hoàn chỉnh"""

        # 1. Tiền xử lý
        processed = self._preprocess(image)

        # 2. Tìm markers
        markers = self._find_markers(processed)

        # 3. Áp dụng watershed
        markers = cv2.watershed(image, markers)

        # 4. Hậu xử lý
        markers = self._postprocess(markers)

        return markers

    def _preprocess(self, image):
        """Tiền xử lý ảnh"""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Khử nhiễu
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Ngưỡng hóa
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Hình thái
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        return binary

    def _find_markers(self, binary):
        """Tìm markers watershed"""

        # Biến đổi khoảng cách
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Nền trước chắc chắn
        _, sure_fg = cv2.threshold(dist, self.dist_threshold * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Nền sau chắc chắn
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)

        # Không xác định
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Gắn nhãn markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        return markers

    def _postprocess(self, markers):
        """Hậu xử lý markers"""

        # Gộp vùng nhỏ
        markers = merge_small_regions(markers, min_size=self.min_area)

        return markers


# Sử dụng
segmenter = AdvancedWatershed(min_area=200, dist_threshold=0.6)

markers = segmenter.segment(image)

# Trực quan hóa
result = image.copy()
result[markers == -1] = [255, 0, 0]

cv2.imshow('Kết Quả Watershed', result)
cv2.waitKey(0)
```

---

**Ngày tạo**: Tháng 1/2025
