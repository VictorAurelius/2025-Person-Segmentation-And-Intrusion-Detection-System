# OpenCV Essentials (Kiến Thức Cơ Bản OpenCV)

## 1. Giới Thiệu OpenCV

### A. Tổng Quan

**OpenCV** (Open Source Computer Vision Library - Thư viện Thị Giác Máy Tính Mã Nguồn Mở) là thư viện mã nguồn mở cho Computer Vision (Thị Giác Máy Tính) và Machine Learning (Học Máy).

**Tính năng chính:**
- 2500+ thuật toán được tối ưu hóa
- Hỗ trợ Python, C++, Java
- Cross-platform (đa nền tảng): Windows, Linux, macOS, Android, iOS
- Real-time processing (xử lý thời gian thực)
- GPU acceleration (tăng tốc GPU) với CUDA

### B. Cài Đặt

```bash
# OpenCV cơ bản
pip install opencv-python

# Với các contrib modules (tính năng mở rộng)
pip install opencv-contrib-python

# Kiểm tra cài đặt
python -c "import cv2; print(cv2.__version__)"
```

### C. Quy Ước Import

```python
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
```

---

## 2. Xử Lý Video

### A. Video Capture (Chụp Video)

#### Từ File

```python
# Mở file video
cap = cv2.VideoCapture('video.mp4')

# Kiểm tra đã mở thành công chưa
if not cap.isOpened():
    print("Lỗi khi mở file video")
    exit()

# Lấy thuộc tính video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Độ phân giải: {width}x{height}")
print(f"Tổng số frames: {frame_count}")
```

#### Từ Webcam

```python
# Mở webcam (0 = camera mặc định)
cap = cv2.VideoCapture(0)

# Đặt độ phân giải
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Đặt FPS
cap.set(cv2.CAP_PROP_FPS, 30)
```

### B. Đọc Frames

```python
while True:
    # Đọc frame
    ret, frame = cap.read()

    # Kiểm tra đã đọc thành công chưa
    if not ret:
        print("Kết thúc video hoặc lỗi")
        break

    # Xử lý frame tại đây
    cv2.imshow('Frame', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
```

### C. Ghi Video

```python
# Định nghĩa codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # hoặc 'XVID', 'MJPG'

# Tạo VideoWriter
out = cv2.VideoWriter(
    'output.mp4',
    fourcc,
    30.0,  # FPS
    (640, 480)  # Độ phân giải (width, height)
)

# Ghi frames
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Xử lý frame
        processed = process_frame(frame)

        # Ghi
        out.write(processed)
    else:
        break

# Giải phóng
out.release()
```

### D. Thao Tác Video Nâng Cao

```python
# Nhảy đến frame cụ thể
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Nhảy đến frame 100

# Lấy vị trí frame hiện tại
current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

# Lấy thời lượng video
duration = frame_count / fps
print(f"Thời lượng: {duration:.2f} giây")

# Bỏ qua frames (để tăng hiệu suất)
skip_frames = 2
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue  # Bỏ qua frame này

    # Xử lý mỗi frame thứ N
    process_frame(frame)
```

---

## 3. Background Subtraction (Trừ Nền)

### A. MOG2

```python
# Tạo MOG2 background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,           # Số frame để học
    varThreshold=16,       # Ngưỡng phát hiện
    detectShadows=True     # Phát hiện và đánh dấu bóng
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Áp dụng background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Hiển thị
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
```

**Tham số:**
- `history`: Lớn hơn = ổn định hơn, thích ứng chậm hơn
- `varThreshold`: Thấp hơn = nhạy hơn
- `detectShadows`: True = đánh dấu bóng (giá trị 127)

### B. KNN

```python
# Tạo KNN background subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)

# Sử dụng giống như MOG2
fg_mask = bg_subtractor.apply(frame)
```

**KNN so với MOG2:**
- KNN: Tốt hơn với nhiễu, chậm hơn
- MOG2: Nhanh hơn, tốt cho hầu hết trường hợp

### C. Ảnh Nền

```python
# Lấy ảnh nền đã học
background = bg_subtractor.getBackgroundImage()

# Hiển thị
cv2.imshow('Learned Background', background)
```

---

## 4. Contour Detection (Phát Hiện Đường Viền)

### A. Tìm Contours

```python
# Chuyển sang ảnh xám
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Ngưỡng hóa
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Tìm contours
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,  # Chế độ truy xuất
    cv2.CHAIN_APPROX_SIMPLE  # Phương pháp xấp xỉ
)

print(f"Tìm thấy {len(contours)} contours")
```

**Retrieval Modes (Chế Độ Truy Xuất):**
- `RETR_EXTERNAL`: Chỉ contours ngoài cùng
- `RETR_LIST`: Tất cả contours, không có hierarchy
- `RETR_TREE`: Hierarchy đầy đủ

**Approximation Methods (Phương Pháp Xấp Xỉ):**
- `CHAIN_APPROX_NONE`: Tất cả điểm
- `CHAIN_APPROX_SIMPLE`: Chỉ các điểm góc (tiết kiệm bộ nhớ)

### B. Vẽ Contours

```python
# Vẽ tất cả contours
cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

# Vẽ contour cụ thể
cv2.drawContours(frame, contours, 0, (0, 255, 0), 2)  # Vẽ contour đầu tiên

# Vẽ tô đầy
cv2.drawContours(frame, contours, -1, (0, 255, 0), -1)
```

### C. Thuộc Tính Contour

```python
for contour in contours:
    # Diện tích
    area = cv2.contourArea(contour)

    # Chu vi
    perimeter = cv2.arcLength(contour, closed=True)

    # Bounding rectangle (hình chữ nhật bao)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Minimum area rectangle (hình chữ nhật xoay)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

    # Centroid (tâm)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Convex hull (bao lồi)
    hull = cv2.convexHull(contour)

    # Aspect ratio (tỷ lệ khung hình)
    aspect_ratio = float(w) / h

    # Extent (tỷ lệ diện tích contour so với diện tích bounding box)
    rect_area = w * h
    extent = float(area) / rect_area

    # Solidity (tỷ lệ diện tích contour so với diện tích convex hull)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
```

### D. Lọc Contour

```python
# Lọc theo diện tích
min_area = 500
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# Lọc theo aspect ratio (cho vật thể dọc như người)
def filter_by_aspect_ratio(contours, min_ratio=0.3, max_ratio=3.0):
    filtered = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        if min_ratio <= aspect_ratio <= max_ratio:
            filtered.append(c)
    return filtered

# Lọc theo solidity (vật thể chặt)
def filter_by_solidity(contours, min_solidity=0.8):
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity >= min_solidity:
            filtered.append(c)
    return filtered
```

---

## 5. Thao Tác ROI

### A. Point in Polygon (Điểm Trong Đa Giác)

```python
# Định nghĩa ROI polygon
roi_points = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])

# Kiểm tra xem điểm có nằm bên trong không
point = (200, 200)
result = cv2.pointPolygonTest(roi_points, point, False)

if result >= 0:
    print("Điểm nằm trong ROI")
else:
    print("Điểm nằm ngoài ROI")
```

### B. Tạo Mask

```python
# Tạo mask từ polygon
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [roi_points], 255)

# Áp dụng mask lên ảnh
masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
```

### C. Tính Chồng Lấn ROI

```python
def calculate_overlap(contour, roi_polygon):
    """Tính phần trăm chồng lấn giữa contour và ROI"""
    # Tạo masks
    h, w = 1080, 1920  # Kích thước frame
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    # Tô đầy masks
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    cv2.fillPoly(roi_mask, [roi_polygon], 255)

    # Tính giao điểm
    intersection = cv2.bitwise_and(contour_mask, roi_mask)
    intersection_area = cv2.countNonZero(intersection)

    # Tính diện tích contour
    contour_area = cv2.contourArea(contour)

    # Phần trăm chồng lấn
    if contour_area > 0:
        overlap = intersection_area / contour_area
    else:
        overlap = 0

    return overlap
```

---

## 6. Tối Ưu Hiệu Suất

### A. Giảm Độ Phân Giải

```python
# Resize để xử lý
scale = 0.5
small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

# Xử lý frame nhỏ hơn
processed_small = process(small_frame)

# Resize lại nếu cần
processed = cv2.resize(processed_small, (frame.shape[1], frame.shape[0]))
```

### B. Xử Lý ROI

```python
# Chỉ xử lý vùng ROI
x, y, w, h = 100, 100, 400, 300
roi = frame[y:y+h, x:x+w]

# Xử lý ROI
processed_roi = process(roi)

# Đặt lại
frame[y:y+h, x:x+w] = processed_roi
```

### C. Bỏ Qua Frames

```python
frame_counter = 0
process_every_n_frames = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % process_every_n_frames == 0:
        # Xử lý frame này
        processed = process(frame)
    else:
        # Sử dụng kết quả trước đó
        pass

    cv2.imshow('Frame', frame)
```

### D. Sử Dụng Ảnh Xám

```python
# Chuyển sang ảnh xám sớm
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Xử lý ảnh xám (nhanh hơn 3 lần so với màu)
processed = process_grayscale(gray)
```

---

## 7. Công Cụ Hữu Ích

### A. Bộ Đếm FPS

```python
import time

class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def update(self):
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        return fps

    def reset(self):
        self.start_time = time.time()
        self.frame_count = 0

# Sử dụng
fps_counter = FPSCounter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý frame
    fps_counter.update()

    # Hiển thị FPS
    fps = fps_counter.get_fps()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
```

### B. Thanh Tiến Trình

```python
def process_video_with_progress(video_path):
    """Xử lý video với hiển thị tiến trình"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý frame
        processed = process(frame)

        # Cập nhật tiến trình
        frame_number += 1
        progress = (frame_number / total_frames) * 100

        print(f"\rTiến trình: {progress:.1f}% ({frame_number}/{total_frames})", end='')

    print("\nHoàn thành!")
    cap.release()
```

### C. Tự Động Khởi Động Lại Khi Lỗi

```python
def robust_capture(source):
    """Video capture mạnh mẽ với tự động khởi động lại"""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Không thể mở. Thử lại {retry_count + 1}/{max_retries}")
            retry_count += 1
            time.sleep(1)
            continue

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Mất kết nối. Đang kết nối lại...")
                break

            # Xử lý frame
            yield frame

        cap.release()
        retry_count += 1

    print("Đã đạt số lần thử tối đa. Thoát.")
```

---

## 8. Mouse Callbacks (Xử Lý Sự Kiện Chuột)

### A. Callback Cơ Bản

```python
def mouse_callback(event, x, y, flags, param):
    """Xử lý sự kiện chuột"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Click trái tại ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Click phải tại ({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        print(f"Chuột di chuyển đến ({x}, {y})")

# Đặt callback
cv2.namedWindow('Window')
cv2.setMouseCallback('Window', mouse_callback)

while True:
    cv2.imshow('Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### B. Chọn ROI Tương Tác

```python
class ROISelector:
    def __init__(self):
        self.points = []
        self.drawing = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.drawing = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Hoàn thành ROI
            if len(self.points) >= 3:
                print(f"ROI hoàn thành với {len(self.points)} điểm")
                self.drawing = False

    def draw_roi(self, frame):
        """Vẽ ROI hiện tại lên frame"""
        if len(self.points) > 0:
            # Vẽ điểm
            for point in self.points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

            # Vẽ đường thẳng
            if len(self.points) > 1:
                pts = np.array(self.points, np.int32)
                cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

        return frame

# Sử dụng
selector = ROISelector()
cv2.namedWindow('Select ROI')
cv2.setMouseCallback('Select ROI', selector.mouse_callback)

while True:
    frame_copy = frame.copy()
    frame_copy = selector.draw_roi(frame_copy)

    cv2.imshow('Select ROI', frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 9. Trackbars (Thanh Trượt)

```python
def nothing(x):
    """Dummy callback cho trackbars"""
    pass

# Tạo cửa sổ với trackbars
cv2.namedWindow('Controls')

# Thêm trackbars
cv2.createTrackbar('Threshold', 'Controls', 127, 255, nothing)
cv2.createTrackbar('Blur', 'Controls', 5, 50, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lấy giá trị trackbar
    threshold_value = cv2.getTrackbarPos('Threshold', 'Controls')
    blur_value = cv2.getTrackbarPos('Blur', 'Controls')

    # Làm cho blur value là số lẻ
    if blur_value % 2 == 0:
        blur_value += 1

    # Áp dụng xử lý
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imshow('Result', binary)
    cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 10. Mẫu Thông Dụng

### A. Template Xử Lý Video

```python
def process_video(input_path, output_path=None):
    """Template tiêu chuẩn cho xử lý video"""
    # Mở video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Lỗi khi mở video")
        return

    # Lấy thuộc tính
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo writer nếu lưu
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Vòng lặp xử lý
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # **XỬ LÝ CỦA BẠN Ở ĐÂY**
            processed_frame = your_processing_function(frame)

            # Lưu nếu writer tồn tại
            if out:
                out.write(processed_frame)

            # Hiển thị
            cv2.imshow('Processing', processed_frame)

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nBị gián đoạn bởi người dùng")

    finally:
        # Dọn dẹp
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        print(f"Đã xử lý {frame_count} frames")
```

### B. Hiển Thị Nhiều Cửa Sổ

```python
def show_multiple_views(frame, fg_mask, edges):
    """Hiển thị nhiều view trong layout có tổ chức"""
    # Resize để hiển thị
    h, w = 300, 400

    frame_resized = cv2.resize(frame, (w, h))
    mask_resized = cv2.resize(fg_mask, (w, h))
    edges_resized = cv2.resize(edges, (w, h))

    # Chuyển ảnh xám sang BGR để stack
    mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)

    # Stack theo chiều ngang
    top_row = np.hstack([frame_resized, mask_bgr])
    bottom_row = np.hstack([edges_bgr, np.zeros((h, w, 3), dtype=np.uint8)])

    # Stack theo chiều dọc
    combined = np.vstack([top_row, bottom_row])

    # Thêm nhãn
    cv2.putText(combined, 'Original', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Foreground', (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Edges', (10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return combined
```

---

**Ngày tạo**: Tháng 1/2025
