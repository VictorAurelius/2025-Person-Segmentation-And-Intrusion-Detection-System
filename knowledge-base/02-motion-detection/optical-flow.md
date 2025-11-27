# Optical Flow (Dòng Quang Học)

## 1. Khái Niệm

**Optical Flow (dòng quang học)** là pattern (mẫu) của apparent motion (chuyển động biểu kiến) của objects (vật thể), surfaces (bề mặt), và edges (cạnh) trong visual scene (cảnh trực quan), được gây ra bởi relative motion (chuyển động tương đối) giữa observer (người quan sát - camera) và scene (cảnh).

### A. Định Nghĩa

Optical flow tính toán **motion vector field (trường vector chuyển động)** giữa 2 frames:

```
V(x, y) = (dx, dy)
```

Trong đó:
- `V(x, y)`: Velocity vector (vector vận tốc) tại pixel (x, y)
- `dx`: Displacement (độ dịch chuyển) theo trục x
- `dy`: Displacement (độ dịch chuyển) theo trục y

### B. Giả Định

Optical flow dựa trên 3 giả định:

1. **Brightness Constancy (Độ sáng không đổi)**: Pixel intensity (cường độ pixel) không đổi giữa frames
   ```
   I(x, y, t) = I(x + dx, y + dy, t + dt)
   ```

2. **Temporal Continuity (Liên tục thời gian)**: Chuyển động thay đổi mượt mà theo thời gian

3. **Spatial Coherence (Tính mạch lạc không gian)**: Các pixel lân cận thuộc cùng một bề mặt và có chuyển động tương tự

---

## 2. Lucas-Kanade Method (Phương Pháp Lucas-Kanade)

### A. Giới Thiệu

Lucas-Kanade là sparse (thưa) optical flow method, tính motion vectors (vector chuyển động) tại specific points (điểm cụ thể - features) thay vì toàn bộ image.

**Paper:** "An Iterative Image Registration Technique" - Lucas & Kanade (1981)

### B. Nguyên Lý

Giả sử motion là constant (không đổi) trong local neighborhood (vùng lân cận cục bộ):

```
Optical Flow Equation (Phương trình Optical Flow):
I_x * u + I_y * v + I_t = 0

Trong đó:
- I_x, I_y: Spatial image gradients (gradient ảnh không gian)
- I_t: Temporal gradient (gradient thời gian)
- u, v: Flow vectors (vector dòng - dx/dt, dy/dt)
```

### C. Triển Khai

```python
import cv2
import numpy as np

# Tham số cho phát hiện góc ShiTomasi
feature_params = dict(
    maxCorners=100,      # Số features tối đa
    qualityLevel=0.3,    # Quality threshold
    minDistance=7,       # Khoảng cách tối thiểu giữa corners
    blockSize=7          # Size of averaging block
)

# Tham số cho optical flow Lucas-Kanade
lk_params = dict(
    winSize=(15, 15),           # Window size
    maxLevel=2,                 # Pyramid levels
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,    # Max iterations
        0.03   # Epsilon
    )
)

# Đọc frame đầu tiên
cap = cv2.VideoCapture('video.mp4')
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Phát hiện các đặc trưng ban đầu
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Tạo mask để vẽ
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Tính optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    # Chọn điểm tốt
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Vẽ quỹ đạo
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        # Vẽ đường (quỹ đạo chuyển động)
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
                       (0, 255, 0), 2)

        # Vẽ điểm
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Kết hợp frame và mask
    result = cv2.add(frame, mask)

    cv2.imshow('Optical Flow', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Cập nhật frame và điểm trước
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
```

### D. Tham Số

#### winSize

```python
# Small window: Fast, less robust
lk_params = dict(winSize=(7, 7), ...)

# Medium window: Balanced (recommended)
lk_params = dict(winSize=(15, 15), ...)

# Large window: Slower, more robust
lk_params = dict(winSize=(31, 31), ...)
```

#### maxLevel

```python
# No pyramid: Fast, less accurate
lk_params = dict(maxLevel=0, ...)

# 2-3 levels: Good balance (recommended)
lk_params = dict(maxLevel=2, ...)

# Many levels: Handle large motions
lk_params = dict(maxLevel=4, ...)
```

---

## 3. Farneback Method (Phương Pháp Farneback)

### A. Giới Thiệu

Farneback là dense (dày đặc) optical flow method, tính motion vectors cho mọi pixel trong image.

**Paper:** "Two-Frame Motion Estimation Based on Polynomial Expansion" - Farneback (2003)

### B. Implementation

```python
def draw_flow(img, flow, step=16):
    """Vẽ các vector optical flow lên ảnh"""
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Tạo các đường
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Vẽ
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    # Vẽ điểms
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

# Đọc frame đầu tiên
cap = cv2.VideoCapture('video.mp4')
ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Tính dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,    # Pyramid scale
        levels=3,          # Pyramid levels
        winsize=15,        # Window size
        iterations=3,      # Iterations at each level
        poly_n=5,          # Polynomial expansion neighborhood
        poly_sigma=1.2,    # Gaussian std for polynomial expansion
        flags=0
    )

    # Trực quan hóa
    result = draw_flow(gray, flow)

    cv2.imshow('Dense Optical Flow', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
```

### C. Flow Visualization

#### Vector Field

```python
def draw_flow_vectors(img, flow, step=16):
    """Draw flow as vector field"""
    h, w = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]

            # Vẽ arrow
            cv2.arrowedLine(
                vis,
                (x, y),
                (int(x + fx), int(y + fy)),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    return vis
```

#### HSV Representation

```python
def flow_to_hsv(flow):
    """Convert flow to HSV image"""
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    # Tính độ lớn and angle
    mag, ang = cv2.cartToPolar(fx, fy)

    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
    hsv[..., 1] = 255                     # Saturation = full
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude

    # Convert to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr
```

---

## 4. Phát Hiện Chuyển Động Từ Flow

### A. Ngưỡng Độ Lớn

```python
def detect_motion_from_flow(flow, threshold=2.0):
    """Phát hiện chuyển động từ optical flow"""
    # Tính độ lớn
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.sqrt(fx**2 + fy**2)

    # Ngưỡng
    motion_mask = (magnitude > threshold).astype(np.uint8) * 255

    return motion_mask
```

### B. Lọc Theo Hướng

```python
def detect_motion_by_direction(flow, direction='right', threshold=2.0):
    """Phát hiện chuyển động theo hướng cụ thể"""
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.sqrt(fx**2 + fy**2)

    # Tính góc
    angle = np.arctan2(fy, fx) * 180 / np.pi

    # Phạm vi hướng (độ)
    directions = {
        'right': (- 45, 45),
        'down': (45, 135),
        'left': (135, 180) or (-180, -135),
        'up': (-135, -45)
    }

    # Lọc theo hướng
    if direction in directions:
        min_ang, max_ang = directions[direction]
        direction_mask = (angle >= min_ang) & (angle <= max_ang)
        motion_mask = direction_mask & (magnitude > threshold)
        motion_mask = motion_mask.astype(np.uint8) * 255

        return motion_mask

    return None
```

---

## 5. Ứng Dụng

### A. Theo Dõi Vật Thể

```python
class OpticalFlowTracker:
    """Theo dõi vật thể sử dụng optical flow"""

    def __init__(self):
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.tracks = []
        self.track_len = 10

    def update(self, frame, roi=None):
        """Cập nhật theo dõi với frame mới"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(self.tracks) > 0:
            # Tính optical flow
            img0, img1 = self.prev_gray, gray
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)

            # Calculate error
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1

            # Update tracks
            new_tracks = []
            for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue

                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]

                new_tracks.append(tr)

            self.tracks = new_tracks

        # Detect new features
        if len(self.tracks) < 100:
            mask = np.zeros_like(gray)
            mask[:] = 255

            # Mask existing tracks
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            # Detect new points
            p = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)

            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])

        self.prev_gray = gray

        return self.tracks

    def draw_tracks(self, frame):
        """Vẽ quỹ đạo lên frame"""
        vis = frame.copy()

        for tr in self.tracks:
            # Vẽ trail
            pts = np.int32(tr)
            cv2.polylines(vis, [pts], False, (0, 255, 0))

            # Vẽ current point
            if len(tr) > 0:
                x, y = np.int32(tr[-1])
                cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        return vis
```

### B. Phân Vùng Chuyển Động

```python
def segment_moving_objects(flow, threshold=2.0, min_area=500):
    """Phân vùng vật thể chuyển động từ flow"""
    # Lấy motion mask
    motion_mask = detect_motion_from_flow(flow, threshold)

    # Phép toán hình thái
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

    # Tìm contours
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc theo diện tích
    objects = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            objects.append({'bbox': (x, y, w, h), 'contour': contour})

    return objects
```

---

## 6. So Sánh: Lucas-Kanade vs Farneback

| Aspect | Lucas-Kanade | Farneback |
|--------|--------------|-----------|
| Type | Sparse | Dense |
| Speed | Fast ✅ | Slow ⚠️ |
| Coverage | Features only | All pixels ✅ |
| Accuracy | High at features ✅ | Medium |
| Use Case | Tracking specific points | Motion fields |
| Memory | Low ✅ | High ⚠️ |

**Khi nào dùng:**

**Lucas-Kanade:**
- Theo dõi vật thể cụ thể
- Cần hiệu năng thời gian thực
- Thông tin chuyển động thưa là đủ

**Farneback:**
- Cần trường chuyển động đầy đủ
- Phân vùng chuyển động
- Hiệu ứng hình ảnh / phân tích

---

## 7. Tối Ưu Hiệu Năng

### A. Giảm Độ Phân Giải

```python
# Giảm mẫu cho optical flow
scale = 0.5
small_prev = cv2.resize(prev_gray, None, fx=scale, fy=scale)
small_curr = cv2.resize(curr_gray, None, fx=scale, fy=scale)

# Tính flow on small images
flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None, ...)

# Scale flow trở lại
flow = flow / scale
flow = cv2.resize(flow, (curr_gray.shape[1], curr_gray.shape[0]))
```

### B. Xử Lý ROI

```python
# Định nghĩa ROI
x, y, w, h = 100, 100, 400, 300

roi_prev = prev_gray[y:y+h, x:x+w]
roi_curr = curr_gray[y:y+h, x:x+w]

# Tính flow chỉ trong ROI
flow_roi = cv2.calcOpticalFlowFarneback(roi_prev, roi_curr, None, ...)
```

### C. Bỏ Qua Frames

```python
# Tính flow mỗi N frames
frame_count = 0
flow_every_n = 3
cached_flow = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1
    if frame_count % flow_every_n == 0:
        if prev_gray is not None:
            cached_flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, ...)

    # Dùng flow đã cache
    if cached_flow is not None:
        # Process with cached_flow
        pass

    prev_gray = gray
```

---

## 8. Hạn Chế

### A. Vi Phạm Độ Sáng Không Đổi

**Problem:**
- Lighting changes
- Shadows
- Occlusions

**Solution:**
```python
# Chuẩn hóa độ chiếu sáng
prev_normalized = cv2.equalizeHist(prev_gray)
curr_normalized = cv2.equalizeHist(curr_gray)

flow = cv2.calcOpticalFlowFarneback(prev_normalized, curr_normalized, ...)
```

### B. Chuyển Động Lớn

**Problem:** Assumptions break down with large displacements

**Solution:**
```python
# Dùng nhiều cấp pyramid hơn
flow = cv2.calcOpticalFlowFarneback(
    prev_gray,
    curr_gray,
    None,
    pyr_scale=0.5,
    levels=5,  # More levels for large motion
    ...
)
```

### C. Ranh Giới Chuyển Động

**Problem:** Inaccurate at object boundaries

**Solution:**
```python
# Dùng kích thước cửa sổ nhỏ hơn tại ranh giới
# Hoặc dùng optical flow nhận biết cạnh (nâng cao)
```

---

## 9. Nâng Cao: DIS Optical Flow

### A. Tìm Kiếm Nghịch Đảo Dày Đặc

```python
# Tạo DIS optical flow
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

# Tính flow
flow = dis.calc(prev_gray, curr_gray, None)

# Các preset:
# - DISOPTICAL_FLOW_PRESET_ULTRAFAST
# - DISOPTICAL_FLOW_PRESET_FAST
# - DISOPTICAL_FLOW_PRESET_MEDIUM (recommended)
```

**Ưu điểm:**
- Nhanh hơn Farneback
- Tốt hơn trong việc giữ ranh giới
- Độ chính xác tốt

---

**Ngày tạo**: Tháng 1/2025
