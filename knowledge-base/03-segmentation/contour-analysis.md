# Contour Analysis (Phân Tích Đường Viền)

## 1. Khái Niệm

**Contour (đường viền)** là đường cong nối các điểm liên tục có cùng màu sắc hoặc cường độ sáng. Trong ngữ cảnh của OpenCV, contours là đường biên của các đối tượng.

### A. Định Nghĩa

```python
# Contour: Danh sách các điểm định nghĩa ranh giới
contour = np.array([
    [[100, 100]],
    [[150, 100]],
    [[150, 150]],
    [[100, 150]]
])  # Shape: (N, 1, 2)
```

---

## 2. Tìm Đường Viền

### A. Sử Dụng Cơ Bản

```python
import cv2
import numpy as np

# Đọc và chuẩn bị ảnh
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ngưỡng hóa để có ảnh nhị phân
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Tìm đường viền
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,        # Chế độ truy xuất
    cv2.CHAIN_APPROX_SIMPLE   # Phương pháp xấp xỉ
)

print(f"Tìm thấy {len(contours)} đường viền")
```

### B. Chế Độ Truy Xuất

```python
# RETR_EXTERNAL: Chỉ lấy đường viền ngoài cùng
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# RETR_LIST: Tất cả đường viền, không có phân cấp
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# RETR_TREE: Tất cả đường viền với phân cấp đầy đủ
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# RETR_CCOMP: Phân cấp 2 mức
contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
```

**Định Dạng Hierarchy (Phân Cấp):**
```
hierarchy[i] = [next, previous, first_child, parent]
```

### C. Phương Pháp Xấp Xỉ

```python
# CHAIN_APPROX_NONE: Lưu tất cả các điểm
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"Điểm: {len(contours[0])}")  # Nhiều điểm

# CHAIN_APPROX_SIMPLE: Chỉ lưu điểm góc (khuyến nghị)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Điểm: {len(contours[0])}")  # Ít điểm hơn
```

---

## 3. Vẽ Đường Viền

### A. Vẽ Cơ Bản

```python
# Vẽ tất cả đường viền
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Vẽ đường viền cụ thể
cv2.drawContours(result, contours, 0, (0, 255, 0), 2)  # Đường viền đầu tiên

# Vẽ tô đầy
cv2.drawContours(result, contours, -1, (0, 255, 0), -1)  # thickness=-1
```

### B. Vẽ Tùy Chỉnh

```python
def draw_contours_custom(image, contours):
    """Vẽ đường viền với nhiều màu khác nhau"""
    result = image.copy()

    for i, contour in enumerate(contours):
        # Màu ngẫu nhiên cho mỗi đường viền
        color = np.random.randint(0, 255, 3).tolist()

        # Vẽ đường viền
        cv2.drawContours(result, [contour], -1, color, 2)

        # Vẽ chỉ số đường viền
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result, str(i), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result
```

---

## 4. Thuộc Tính Đường Viền

### A. Diện Tích

```python
for contour in contours:
    area = cv2.contourArea(contour)
    print(f"Diện tích: {area} pixels²")
```

### B. Chu Vi

```python
for contour in contours:
    # Đường viền đóng
    perimeter = cv2.arcLength(contour, closed=True)
    print(f"Chu vi: {perimeter} pixels")

    # Đường viền mở
    perimeter_open = cv2.arcLength(contour, closed=False)
```

### C. Bounding Rectangle (Hình Chữ Nhật Bao)

```python
for contour in contours:
    # Hình chữ nhật bao thẳng
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"BBox: ({x}, {y}, {w}, {h})")
```

### D. Rotated Rectangle (Hình Chữ Nhật Xoay)

```python
for contour in contours:
    # Hình chữ nhật diện tích tối thiểu (có thể xoay)
    rect = cv2.minAreaRect(contour)
    # rect = ((center_x, center_y), (width, height), angle)

    # Lấy tọa độ các điểm góc
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Vẽ
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
```

### E. Minimum Enclosing Circle (Hình Tròn Bao Nhỏ Nhất)

```python
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    cv2.circle(image, center, radius, (255, 0, 0), 2)
```

### F. Centroid (Trọng Tâm)

```python
for contour in contours:
    M = cv2.moments(contour)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
        print(f"Trọng tâm: ({cx}, {cy})")
```

### G. Convex Hull (Bao Lồi)

```python
for contour in contours:
    hull = cv2.convexHull(contour)

    # Vẽ bao lồi
    cv2.drawContours(image, [hull], 0, (0, 255, 255), 2)
```

---

## 5. Xấp Xỉ Đường Viền

### A. Thuật Toán Douglas-Peucker

```python
for contour in contours:
    # Xấp xỉ đường viền với ít điểm hơn
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    print(f"Điểm ban đầu: {len(contour)}")
    print(f"Điểm xấp xỉ: {len(approx)}")

    # Vẽ xấp xỉ
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
```

**Tham Số Epsilon:**
```python
# Epsilon nhỏ: Chi tiết hơn (nhiều điểm)
epsilon = 0.001 * cv2.arcLength(contour, True)

# Epsilon trung bình: Cân bằng (khuyến nghị)
epsilon = 0.01 * cv2.arcLength(contour, True)

# Epsilon lớn: Đơn giản (ít điểm)
epsilon = 0.1 * cv2.arcLength(contour, True)
```

### B. Nhận Dạng Hình Dạng

```python
def detect_shape(contour):
    """Nhận dạng hình dạng từ đường viền"""

    # Xấp xỉ
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Số đỉnh
    vertices = len(approx)

    if vertices == 3:
        return "Tam giác"
    elif vertices == 4:
        # Kiểm tra hình vuông hay chữ nhật
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        if 0.95 <= aspect_ratio <= 1.05:
            return "Hình vuông"
        else:
            return "Hình chữ nhật"
    elif vertices == 5:
        return "Ngũ giác"
    elif vertices > 5:
        # Kiểm tra hình tròn
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius ** 2

        if abs(area - circle_area) / circle_area < 0.2:
            return "Hình tròn"
        else:
            return "Hình elip"
    else:
        return "Không xác định"
```

---

## 6. Đặc Trưng Đường Viền

### A. Aspect Ratio (Tỷ Lệ Khung Hình)

```python
x, y, w, h = cv2.boundingRect(contour)
aspect_ratio = float(w) / h

print(f"Tỷ lệ khung hình: {aspect_ratio}")

# Phân loại
if aspect_ratio > 2:
    print("Đối tượng ngang")
elif aspect_ratio < 0.5:
    print("Đối tượng dọc")
else:
    print("Đối tượng cân bằng")
```

### B. Extent (Độ Mở Rộng)

**Extent** = Diện tích đường viền / Diện tích hình chữ nhật bao

```python
area = cv2.contourArea(contour)
x, y, w, h = cv2.boundingRect(contour)
rect_area = w * h
extent = float(area) / rect_area

print(f"Extent: {extent:.2f}")

# Giải thích:
# extent ≈ 1.0: Lấp đầy hộp bao (hình chữ nhật)
# extent ≈ 0.79: Hình tròn (π/4)
# extent < 0.5: Hình dạng không đều
```

### C. Solidity (Độ Đặc)

**Solidity** = Diện tích đường viền / Diện tích bao lồi

```python
area = cv2.contourArea(contour)
hull = cv2.convexHull(contour)
hull_area = cv2.contourArea(hull)
solidity = float(area) / hull_area if hull_area > 0 else 0

print(f"Solidity: {solidity:.2f}")

# Giải thích:
# solidity ≈ 1.0: Hình lồi
# solidity < 0.8: Có lõm
```

### D. Equivalent Diameter (Đường Kính Tương Đương)

```python
area = cv2.contourArea(contour)
equi_diameter = np.sqrt(4 * area / np.pi)

print(f"Đường kính tương đương: {equi_diameter:.2f} pixels")
```

### E. Orientation (Hướng)

```python
# Khớp elip (yêu cầu ít nhất 5 điểm)
if len(contour) >= 5:
    ellipse = cv2.fitEllipse(contour)
    (x, y), (MA, ma), angle = ellipse

    print(f"Hướng: {angle:.2f} độ")

    # Vẽ elip
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
```

---

## 7. Lọc Đường Viền

### A. Theo Diện Tích

```python
def filter_by_area(contours, min_area=500, max_area=50000):
    """Lọc đường viền theo diện tích"""
    filtered = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered.append(contour)

    return filtered

# Sử dụng
large_contours = filter_by_area(contours, min_area=1000)
```

### B. Theo Hình Dạng

```python
def filter_by_aspect_ratio(contours, min_ratio=0.5, max_ratio=2.0):
    """Lọc đường viền theo tỷ lệ khung hình"""
    filtered = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        if min_ratio <= aspect_ratio <= max_ratio:
            filtered.append(contour)

    return filtered

# Ví dụ: Lọc đối tượng dọc (người, cột)
vertical_contours = filter_by_aspect_ratio(contours, min_ratio=0.3, max_ratio=0.8)
```

### C. Theo Solidity

```python
def filter_by_solidity(contours, min_solidity=0.8):
    """Lọc bỏ đường viền có lỗ/lõm"""
    filtered = []

    for contour in contours:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity = float(area) / hull_area
            if solidity >= min_solidity:
                filtered.append(contour)

    return filtered

# Lọc chỉ hình dạng đặc
compact_contours = filter_by_solidity(contours, min_solidity=0.9)
```

### D. Lọc Kết Hợp

```python
class ContourFilter:
    """Lọc đường viền theo nhiều tiêu chí"""

    def __init__(self, min_area=500, max_area=50000,
                 min_aspect=0.3, max_aspect=3.0,
                 min_solidity=0.7):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_solidity = min_solidity

    def filter(self, contours):
        """Áp dụng tất cả bộ lọc"""
        filtered = []

        for contour in contours:
            if self._meets_criteria(contour):
                filtered.append(contour)

        return filtered

    def _meets_criteria(self, contour):
        """Kiểm tra đường viền đáp ứng tất cả tiêu chí"""

        # Diện tích
        area = cv2.contourArea(contour)
        if not (self.min_area <= area <= self.max_area):
            return False

        # Tỷ lệ khung hình
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if not (self.min_aspect <= aspect_ratio <= self.max_aspect):
            return False

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
            if solidity < self.min_solidity:
                return False

        return True


# Sử dụng
filter = ContourFilter(min_area=1000, max_area=10000,
                        min_aspect=0.5, max_aspect=2.0,
                        min_solidity=0.8)

filtered_contours = filter.filter(contours)
```

---

## 8. Khớp Đường Viền

### A. Khớp Hình Dạng

```python
# Đường viền mẫu
template = contours[0]

# So sánh với các đường viền khác
for i, contour in enumerate(contours[1:], 1):
    match = cv2.matchShapes(template, contour, cv2.CONTOURS_MATCH_I1, 0)

    print(f"Đường viền {i}: khớp = {match:.4f}")

    # Giá trị thấp = khớp tốt
    if match < 0.1:
        print(f"  → Tương tự mẫu!")
```

**Phương Pháp Khớp:**
- `CONTOURS_MATCH_I1`: Phương pháp Hu moments 1
- `CONTOURS_MATCH_I2`: Phương pháp Hu moments 2
- `CONTOURS_MATCH_I3`: Phương pháp Hu moments 3

---

## 9. Phân Tích Phân Cấp

### A. Quan Hệ Cha-Con

```python
def analyze_hierarchy(contours, hierarchy):
    """Phân tích phân cấp đường viền"""

    # hierarchy[i] = [next, previous, first_child, parent]

    for i, contour in enumerate(contours):
        h = hierarchy[0][i]
        next_idx, prev_idx, child_idx, parent_idx = h

        area = cv2.contourArea(contour)

        print(f"Đường viền {i}:")
        print(f"  Diện tích: {area}")
        print(f"  Cha: {parent_idx}")
        print(f"  Con đầu: {child_idx}")

        if parent_idx == -1:
            print(f"  → Đường viền ngoài")
        else:
            print(f"  → Đường viền trong (lỗ)")
```

### B. Phát Hiện Đối Tượng Có Lỗ

```python
def find_objects_with_holes(contours, hierarchy):
    """Tìm đường viền chứa lỗ"""

    objects_with_holes = []

    for i, contour in enumerate(contours):
        h = hierarchy[0][i]
        child_idx = h[2]

        # Có con = có lỗ
        if child_idx != -1:
            objects_with_holes.append((i, contour))

    return objects_with_holes
```

---

## 10. Phân Tích Nâng Cao

### A. Contour Moments (Mô-men Đường Viền)

```python
def analyze_moments(contour):
    """Phân tích đường viền sử dụng mô-men"""

    M = cv2.moments(contour)

    # Mô-men trung tâm
    mu20 = M['mu20']
    mu02 = M['mu02']
    mu11 = M['mu11']

    # Hướng
    if (mu20 - mu02) != 0:
        theta = 0.5 * np.arctan(2 * mu11 / (mu20 - mu02))
        orientation = np.degrees(theta)
    else:
        orientation = 0

    # Độ lệch tâm
    if M['m00'] > 0:
        a = mu20 / M['m00']
        b = mu02 / M['m00']
        if a > b and b > 0:
            eccentricity = np.sqrt(1 - (b / a))
        else:
            eccentricity = 0
    else:
        eccentricity = 0

    return {
        'orientation': orientation,
        'eccentricity': eccentricity
    }
```

### B. Hu Moments (Mô Tả Hình Dạng)

```python
def calculate_hu_moments(contour):
    """Tính Hu moments để nhận dạng hình dạng"""

    M = cv2.moments(contour)
    hu_moments = cv2.HuMoments(M)

    # Biến đổi log để có phạm vi tốt hơn
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

    return hu_moments.flatten()

# So sánh hình dạng
hu1 = calculate_hu_moments(contour1)
hu2 = calculate_hu_moments(contour2)

distance = np.linalg.norm(hu1 - hu2)
print(f"Khác biệt hình dạng: {distance}")
```

---

## 11. Độ Lồi Đường Viền

### A. Kiểm Tra Độ Lồi

```python
is_convex = cv2.isContourConvex(contour)

if is_convex:
    print("Đường viền lồi")
else:
    print("Đường viền có lõm")
```

### B. Convexity Defects (Khuyết Lồi)

```python
# Lấy chỉ số bao lồi
hull = cv2.convexHull(contour, returnPoints=False)

# Tính khuyết
defects = cv2.convexityDefects(contour, hull)

if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # d là khoảng cách từ điểm xa nhất tới bao lồi (trong 1/256 pixels)
        depth = d / 256.0

        # Vẽ khuyết
        cv2.line(image, start, end, (0, 255, 0), 2)
        cv2.circle(image, far, 5, (0, 0, 255), -1)

        print(f"Độ sâu khuyết: {depth:.2f}")
```

---

## 12. Ví Dụ Hoàn Chỉnh

```python
class ContourAnalyzer:
    """Phân tích đường viền toàn diện"""

    def __init__(self, min_area=500):
        self.min_area = min_area

    def analyze(self, image):
        """Phân tích tất cả đường viền trong ảnh"""

        # Chuẩn bị ảnh
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Tìm đường viền
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Phân tích từng đường viền
        results = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            # Tính các thuộc tính
            props = self._calculate_properties(contour)
            props['id'] = i
            props['contour'] = contour

            results.append(props)

        return results

    def _calculate_properties(self, contour):
        """Tính tất cả thuộc tính đường viền"""

        # Thuộc tính cơ bản
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Hộp bao
        x, y, w, h = cv2.boundingRect(contour)

        # Trọng tâm
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Tỷ lệ khung hình
        aspect_ratio = float(w) / h if h > 0 else 0

        # Extent
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Hình dạng
        shape = detect_shape(contour)

        return {
            'area': area,
            'perimeter': perimeter,
            'bbox': (x, y, w, h),
            'centroid': (cx, cy),
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity,
            'shape': shape
        }

    def visualize(self, image, results):
        """Trực quan hóa kết quả phân tích"""

        vis = image.copy()

        for props in results:
            contour = props['contour']
            x, y, w, h = props['bbox']
            cx, cy = props['centroid']

            # Vẽ đường viền
            cv2.drawContours(vis, [contour], 0, (0, 255, 0), 2)

            # Vẽ hộp bao
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Vẽ trọng tâm
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # Thêm nhãn
            label = f"{props['shape']} ({props['area']:.0f})"
            cv2.putText(vis, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis


# Sử dụng
analyzer = ContourAnalyzer(min_area=500)

# Phân tích
results = analyzer.analyze(image)

# In kết quả
for props in results:
    print(f"Đối tượng {props['id']}:")
    print(f"  Hình dạng: {props['shape']}")
    print(f"  Diện tích: {props['area']:.0f}")
    print(f"  Tỷ lệ khung hình: {props['aspect_ratio']:.2f}")
    print(f"  Solidity: {props['solidity']:.2f}")

# Trực quan hóa
vis = analyzer.visualize(image, results)
cv2.imshow('Phân tích', vis)
cv2.waitKey(0)
```

---

**Ngày tạo**: Tháng 1/2025
