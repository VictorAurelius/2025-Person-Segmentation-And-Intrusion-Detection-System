# Background Subtraction (Trừ Nền)

## 1. Khái Niệm

**Background Subtraction (trừ nền)** là kỹ thuật phân tách foreground (vùng tiền cảnh - đối tượng chuyển động) khỏi background (nền tĩnh) bằng cách modeling (mô hình hóa) background và so sánh với frame (khung hình) hiện tại.

### A. Nguyên Lý Cơ Bản

```
Frame Hiện Tại - Background Model = Foreground Mask
```

**Ví dụ:**
```python
# Trừ nền đơn giản
background = first_frame.copy()

while True:
    ret, frame = cap.read()

    # Trừ nền
    diff = cv2.absdiff(frame, background)

    # Ngưỡng hóa để lấy mask
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
```

---

## 2. MOG2 (Mixture of Gaussians 2 - Hỗn Hợp Gaussian 2)

### A. Giới Thiệu

MOG2 là adaptive (thích ứng) background subtraction algorithm (thuật toán trừ nền) dựa trên Gaussian Mixture Model (mô hình hỗn hợp Gaussian). Mỗi pixel được model bằng mixture của K Gaussian distributions (phân phối Gaussian).

**Paper:** "Improved adaptive Gaussian mixture model for background subtraction" - Zivkovic (2004)

### B. Cách Hoạt Động

#### Pixel Modeling (Mô Hình Hóa Pixel)

Mỗi pixel `I(x,y)` được model bằng:

```
P(I(x,y)) = Σ(k=1 to K) w_k * N(μ_k, σ_k²)
```

Trong đó:
- `w_k`: Weight (trọng số) của Gaussian thứ k
- `μ_k`: Mean (giá trị trung bình)
- `σ_k²`: Variance (phương sai)
- `K`: Số Gaussians (thường 3-5)

#### Classification (Phân Loại)

```python
# Mã giả
if pixel matches any background Gaussian:
    pixel = BACKGROUND
else:
    pixel = FOREGROUND
```

### C. Triển Khai

```python
# Tạo MOG2 subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,           # Số frames học background
    varThreshold=16,       # Ngưỡng phát hiện
    detectShadows=True     # Phát hiện bóng
)

# Xử lý video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Áp dụng trừ nền
    fg_mask = bg_subtractor.apply(frame, learningRate=-1)  # -1 = tự động

    # Hậu xử lý
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('Foreground', fg_mask)
```

### D. Tham Số

#### history (mặc định: 500)

```python
# Lịch sử ngắn: Thích ứng nhanh với thay đổi
bg_sub = cv2.createBackgroundSubtractorMOG2(history=200)

# Lịch sử dài: Ổn định hơn, thích ứng chậm hơn
bg_sub = cv2.createBackgroundSubtractorMOG2(history=1000)
```

**Hướng dẫn:**
```
Cảnh thay đổi nhanh: 200-300
Cảnh bình thường: 500-700
Cảnh ổn định: 800-1000
```

#### varThreshold (mặc định: 16)

```python
# Ngưỡng thấp: Nhạy hơn (nhiều false positives hơn)
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=10)

# Ngưỡng cao: Ít nhạy hơn (có thể bỏ lỡ chuyển động chậm)
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=25)
```

**Hiệu ứng:**
```
varThreshold = 10:  Độ nhạy ████████████ (Cao)
varThreshold = 16:  Độ nhạy ████████ (Trung bình)
varThreshold = 25:  Độ nhạy █████ (Thấp)
```

#### detectShadows (mặc định: True)

```python
# Bật phát hiện bóng
bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Giá trị fg_mask:
# 0   = Nền (đen)
# 127 = Bóng (xám)
# 255 = Tiền cảnh (trắng)

# Loại bỏ bóng
fg_mask[fg_mask == 127] = 0  # Coi bóng như nền
```

### E. Sử Dụng Nâng Cao

#### Learning Rate (Tốc Độ Học)

```python
# Tốc độ học tự động (mặc định)
fg_mask = bg_sub.apply(frame, learningRate=-1)

# Tốc độ học thủ công (0.0 đến 1.0)
fg_mask = bg_sub.apply(frame, learningRate=0.01)

# Không học (chỉ dùng mô hình hiện tại)
fg_mask = bg_sub.apply(frame, learningRate=0)
```

**Hiệu ứng Learning Rate:**
```
0.0:  Không cập nhật (mô hình đóng băng)
0.001: Thích ứng rất chậm
0.01:  Thích ứng chậm (khuyên dùng cho cảnh ổn định)
0.1:   Thích ứng nhanh (cho cảnh thay đổi)
1.0:   Cập nhật tức thì (giống frame differencing)
```

#### Lấy/Đặt Background

```python
# Lấy background đã học
background = bg_sub.getBackgroundImage()
cv2.imshow('Learned Background', background)

# Lưu mô hình background (để tái sử dụng)
# Lưu ý: MOG2 không hỗ trợ lưu trực tiếp, cần workaround
```

---

## 3. KNN (K-Nearest Neighbors - K Láng Giềng Gần Nhất)

### A. Giới Thiệu

KNN background subtractor sử dụng K-nearest neighbors (K láng giềng gần nhất) để classify (phân loại) mỗi pixel là background hay foreground.

**Paper:** "Efficient Adaptive Density Estimation per Image Pixel" - Zivkovic & Heijden (2006)

### B. Cách Hoạt Động

```
1. Duy trì tập mẫu cho mỗi pixel
2. So sánh pixel hiện tại với K mẫu gần nhất
3. Nếu khoảng cách < ngưỡng → Nền
4. Ngược lại → Tiền cảnh
```

### C. Triển Khai

```python
# Tạo KNN subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)

# Sử dụng (giống MOG2)
fg_mask = bg_subtractor.apply(frame)
```

### D. Tham Số

#### dist2Threshold (mặc định: 400.0)

```python
# Ngưỡng thấp: Nhạy hơn
bg_sub = cv2.createBackgroundSubtractorKNN(dist2Threshold=200.0)

# Ngưỡng cao: Ít nhạy hơn
bg_sub = cv2.createBackgroundSubtractorKNN(dist2Threshold=600.0)
```

### E. So Sánh KNN vs MOG2

| Khía Cạnh | MOG2 | KNN |
|--------|------|-----|
| Tốc Độ | Nhanh hơn ✅ | Chậm hơn ⚠️ |
| Độ Chính Xác | Tốt | Tốt hơn ✅ |
| Xử Lý Nhiễu | Trung bình | Tốt hơn ✅ |
| Bộ Nhớ | Thấp hơn ✅ | Cao hơn ⚠️ |
| Phát Hiện Bóng | Có | Có |

**Khuyến Nghị:**
- Dùng MOG2 cho mục đích chung (cân bằng tốt nhất)
- Dùng KNN khi xử lý video nhiễu
- Dùng MOG2 nếu tốc độ là quan trọng

---

## 4. Hậu Xử Lý

### A. Morphological Operations (Phép Toán Hình Thái)

```python
def post_process_mask(fg_mask):
    """Làm sạch foreground mask"""
    # Loại bỏ nhiễu (opening)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

    # Lấp lỗ (closing)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

    # Giãn nở để kết nối các vùng gần nhau
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=1)

    return fg_mask
```

### B. Loại Bỏ Bóng

```python
def remove_shadows(fg_mask):
    """Loại bỏ pixel bóng (giá trị 127)"""
    # Coi bóng như nền
    fg_mask[fg_mask == 127] = 0

    # Hoặc chỉ giữ tiền cảnh
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    return fg_mask
```

### C. Lọc Theo Kích Thước

```python
def filter_by_size(fg_mask, min_area=500):
    """Loại bỏ các blob nhỏ"""
    # Tìm contours
    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Tạo mask sạch
    clean_mask = np.zeros_like(fg_mask)

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)

    return clean_mask
```

---

## 5. Ví Dụ Hoàn Chỉnh

```python
class MotionDetector:
    """Bộ phát hiện chuyển động sử dụng background subtraction"""

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
                dist2Threshold=threshold * 25,  # Chuyển sang thang KNN
                detectShadows=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def detect(self, frame):
        """Phát hiện chuyển động trong frame"""
        # Áp dụng trừ nền
        fg_mask = self.bg_sub.apply(frame)

        # Loại bỏ bóng
        fg_mask[fg_mask == 127] = 0

        # Phép toán hình thái
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        return fg_mask

    def get_contours(self, fg_mask, min_area=500):
        """Tìm contours từ foreground mask"""
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Lọc theo diện tích
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

        return filtered

    def get_background(self):
        """Lấy ảnh background đã học"""
        return self.bg_sub.getBackgroundImage()


# Sử dụng
detector = MotionDetector(method='MOG2', history=500, threshold=16)

cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện chuyển động
    fg_mask = detector.detect(frame)

    # Lấy contours
    contours = detector.get_contours(fg_mask, min_area=500)

    # Vẽ kết quả
    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Hiển thị
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Result', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 6. Khắc Phục Sự Cố

### A. Quá Nhiều False Positives

**Vấn đề:** Mọi thứ đều được phát hiện là chuyển động

**Giải pháp:**
```python
# 1. Tăng ngưỡng
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=25)

# 2. Lịch sử dài hơn
bg_sub = cv2.createBackgroundSubtractorMOG2(history=1000)

# 3. Morphology mạnh hơn
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

# 4. Diện tích tối thiểu cao hơn
contours = filter_by_area(contours, min_area=1000)
```

### B. Bỏ Lỡ Chuyển Động Chậm

**Vấn đề:** Vật thể chuyển động chậm không được phát hiện

**Giải pháp:**
```python
# 1. Giảm ngưỡng
bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=10)

# 2. Học chậm hơn
fg_mask = bg_sub.apply(frame, learningRate=0.001)

# 3. Dùng KNN (tốt hơn cho chuyển động chậm)
bg_sub = cv2.createBackgroundSubtractorKNN()
```

### C. Nền Trở Thành Tiền Cảnh

**Vấn đề:** Vật thể tĩnh bị coi là tiền cảnh

**Giải pháp:**
```python
# 1. Lịch sử dài hơn
bg_sub = cv2.createBackgroundSubtractorMOG2(history=1000)

# 2. Để mô hình thích ứng
for _ in range(100):  # Học từ 100 frames
    bg_sub.apply(frame)

# 3. Reset thủ công định kỳ
if frame_count % 1000 == 0:
    bg_sub = cv2.createBackgroundSubtractorMOG2()  # Reset
```

### D. Bóng Được Phát Hiện Như Vật Thể

**Vấn đề:** Bóng gây ra phát hiện sai

**Giải pháp:**
```python
# 1. Bật phát hiện bóng
bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# 2. Loại bỏ bóng khỏi mask
fg_mask[fg_mask == 127] = 0

# 3. Điều chỉnh ngưỡng bóng
bg_sub.setShadowThreshold(0.5)  # Mặc định là 0.5
```

---

## 7. Mẹo Tối Ưu Hiệu Năng

### A. Giảm Độ Phân Giải

```python
# Resize trước khi xử lý
scale = 0.5
small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
fg_mask = bg_sub.apply(small_frame)

# Resize mask trở lại nếu cần
fg_mask = cv2.resize(fg_mask, (frame.shape[1], frame.shape[0]))
```

### B. Chỉ Xử Lý ROI

```python
# Định nghĩa ROI
x, y, w, h = 100, 100, 400, 300
roi = frame[y:y+h, x:x+w]

# Xử lý ROI
fg_mask_roi = bg_sub.apply(roi)

# Mask đầy đủ
fg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
fg_mask[y:y+h, x:x+w] = fg_mask_roi
```

### C. Bỏ Qua Frames

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
        # Xử lý mask...
```

---

## 8. Kỹ Thuật Nâng Cao

### A. Nhiều Mô Hình Background

```python
# Dùng các mô hình khác nhau cho các khoảng thời gian khác nhau
class AdaptiveDetector:
    def __init__(self):
        self.day_model = cv2.createBackgroundSubtractorMOG2(history=500)
        self.night_model = cv2.createBackgroundSubtractorMOG2(history=300)

    def detect(self, frame):
        # Xác định ánh sáng
        brightness = np.mean(frame)

        if brightness > 100:  # Ban ngày
            return self.day_model.apply(frame)
        else:  # Ban đêm
            return self.night_model.apply(frame)
```

### B. Học Có Chọn Lọc

```python
# Chỉ cập nhật background trong các vùng nhất định
def selective_update(bg_sub, frame, static_regions_mask):
    """Cập nhật background chỉ trong các vùng tĩnh"""
    # Lấy foreground không học
    fg_mask = bg_sub.apply(frame, learningRate=0)

    # Tạo mask học (nơi cập nhật)
    learn_mask = static_regions_mask & ~fg_mask

    # Cập nhật chỉ trong vùng học
    fg_mask = bg_sub.apply(frame, learningRate=0.01)

    return fg_mask
```

---

**Ngày tạo**: Tháng 1/2025
