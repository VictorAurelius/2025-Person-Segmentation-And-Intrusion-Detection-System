# Object Tracking (Theo Dõi Đối Tượng)

## 1. Khái Niệm

**Object Tracking (Theo dõi đối tượng)** là quá trình định vị một moving object (đối tượng chuyển động) (hoặc nhiều objects) theo thời gian sử dụng camera.

### A. Tracking vs Detection

| Khía Cạnh | Detection (Phát Hiện) | Tracking (Theo Dõi) |
|-----------|-----------|----------|
| Mục Tiêu | Tìm objects trong single frame | Theo dõi objects qua các frames |
| Tốc Độ | Chậm hơn | Nhanh hơn |
| Tính Nhất Quán | Không có lịch sử | Sử dụng thông tin temporal |
| ID | Không có ID liên tục | Duy trì object ID |

---

## 2. Centroid Tracking (Theo Dõi Tâm Điểm)

### A. Thuật Toán

```
1. Phát hiện objects trong frame
2. Tính toán centroids (tâm điểm)
3. Khớp với centroids trước đó
4. Cập nhật IDs
```

### B. Triển Khai

```python
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    """Simple centroid-based object tracker"""

    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()  # {id: centroid}
        self.disappeared = OrderedDict()  # {id: frames_disappeared}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """Đăng ký object mới"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Hủy đăng ký object"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, bboxes):
        """Cập nhật tracker với detections mới"""

        # Không có detections
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Hủy đăng ký nếu biến mất quá lâu
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # Tính toán centroids từ bboxes
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")

        for (i, (x, y, w, h)) in enumerate(bboxes):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)

        # Không có objects hiện có
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        # Khớp objects hiện có với centroids mới
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Tính toán distance matrix
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Tìm các kết cặp khoảng cách tối thiểu
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Cập nhật object centroid
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Xử lý objects biến mất
            unused_rows = set(range(0, D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Đăng ký objects mới
            unused_cols = set(range(0, D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# Sử dụng
tracker = CentroidTracker(max_disappeared=50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện objects (trả về bounding boxes)
    bboxes = detect_objects(frame)  # [(x, y, w, h), ...]

    # Cập nhật tracker
    objects = tracker.update(bboxes)

    # Vẽ tracked objects
    for (object_id, centroid) in objects.items():
        text = f"ID {object_id}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 3. OpenCV Trackers

### A. Các Trackers Có Sẵn

OpenCV cung cấp nhiều tracking algorithms:

```python
# CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
tracker = cv2.TrackerCSRT_create()

# KCF (Kernelized Correlation Filters)
tracker = cv2.TrackerKCF_create()

# MOSSE (Minimum Output Sum of Squared Error)
tracker = cv2.TrackerMOSSE_create()

# MedianFlow
tracker = cv2.legacy.TrackerMedianFlow_create()

# MIL (Multiple Instance Learning)
tracker = cv2.TrackerMIL_create()
```

### B. Sử Dụng Cơ Bản

```python
# Chọn ROI
bbox = cv2.selectROI('Frame', frame, False)

# Tạo tracker
tracker = cv2.TrackerCSRT_create()

# Khởi tạo
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Cập nhật tracker
    success, bbox = tracker.update(frame)

    if success:
        # Vẽ bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Mat", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### C. Multi-Object Tracking (Theo Dõi Nhiều Đối Tượng)

```python
class MultiTracker:
    """Theo dõi nhiều objects sử dụng OpenCV trackers"""

    def __init__(self, tracker_type='CSRT'):
        self.tracker_type = tracker_type
        self.trackers = []
        self.colors = []

    def add(self, frame, bbox):
        """Thêm tracker mới"""

        # Tạo tracker
        if self.tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        else:
            tracker = cv2.TrackerCSRT_create()

        # Khởi tạo
        tracker.init(frame, bbox)

        # Lưu trữ
        self.trackers.append(tracker)
        self.colors.append(tuple(np.random.randint(0, 255, 3).tolist()))

    def update(self, frame):
        """Cập nhật tất cả trackers"""

        results = []

        for i, tracker in enumerate(self.trackers):
            success, bbox = tracker.update(frame)

            if success:
                results.append({
                    'id': i,
                    'bbox': bbox,
                    'color': self.colors[i]
                })

        return results

    def remove(self, tracker_id):
        """Xóa tracker"""
        if 0 <= tracker_id < len(self.trackers):
            del self.trackers[tracker_id]
            del self.colors[tracker_id]


# Sử dụng
multi_tracker = MultiTracker(tracker_type='CSRT')

# Chọn nhiều ROIs
while True:
    bbox = cv2.selectROI('Select Object', frame, False)
    if bbox[2] == 0 or bbox[3] == 0:
        break

    multi_tracker.add(frame, bbox)

# Vòng lặp tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Cập nhật tất cả trackers
    results = multi_tracker.update(frame)

    # Vẽ kết quả
    for result in results:
        x, y, w, h = [int(v) for v in result['bbox']]
        cv2.rectangle(frame, (x, y), (x + w, y + h), result['color'], 2)

        text = f"ID {result['id']}"
        cv2.putText(frame, text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, result['color'], 2)

    cv2.imshow('Multi-Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 4. Kalman Filter Tracking

### A. Khái Niệm

Kalman Filter dự đoán vị trí object và điều chỉnh dự đoán sử dụng measurements (đo đạc).

### B. Triển Khai

```python
class KalmanTracker:
    """Kalman filter-based object tracker"""

    def __init__(self):
        # Tạo Kalman filter
        # State: [x, y, dx, dy] (vị trí + vận tốc)
        self.kalman = cv2.KalmanFilter(4, 2)

        # Transition matrix
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

    def predict(self):
        """Dự đoán vị trí tiếp theo"""
        prediction = self.kalman.predict()
        return prediction[:2].flatten()  # Trả về [x, y]

    def correct(self, measurement):
        """Điều chỉnh dự đoán với measurement"""
        measurement = np.array([[np.float32(measurement[0])],
                               [np.float32(measurement[1])]])
        corrected = self.kalman.correct(measurement)
        return corrected[:2].flatten()


# Sử dụng
tracker = KalmanTracker()

# Khởi tạo với detection đầu tiên
first_detection = (100, 100)  # (x, y)
tracker.kalman.statePost = np.array([
    [first_detection[0]],
    [first_detection[1]],
    [0],
    [0]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán
    predicted = tracker.predict()

    # Phát hiện object (trả về centroid hoặc None)
    detection = detect_object(frame)

    if detection is not None:
        # Điều chỉnh với measurement
        corrected = tracker.correct(detection)

        # Vẽ detection (đỏ) và corrected (xanh lá)
        cv2.circle(frame, tuple(detection.astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(corrected.astype(int)), 5, (0, 255, 0), -1)
    else:
        # Không có detection, chỉ dùng dự đoán
        cv2.circle(frame, tuple(predicted.astype(int)), 5, (255, 0, 0), -1)

    cv2.imshow('Kalman Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 5. DeepSORT-inspired Tracking

### A. DeepSORT Đơn Giản Hóa

```python
from scipy.spatial import distance as dist

class Track:
    """Single track"""

    def __init__(self, track_id, bbox, feature=None):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 0
        self.state = 'tentative'  # tentative hoặc confirmed

    def update(self, bbox, feature=None):
        """Cập nhật track với detection mới"""
        self.bbox = bbox
        if feature is not None:
            self.feature = feature
        self.age += 1
        self.time_since_update = 0
        self.hit_streak += 1

        if self.hit_streak >= 3:
            self.state = 'confirmed'

    def mark_missed(self):
        """Đánh dấu track bị missed"""
        self.time_since_update += 1
        self.hit_streak = 0


class SimpleDeepSORT:
    """Simplified DeepSORT tracker"""

    def __init__(self, max_age=30, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """
        Cập nhật tracks với detections mới

        detections: list of (bbox, feature)
        """

        # Dự đoán (trong DeepSORT đầy đủ, sử dụng Kalman filter ở đây)

        # Khớp detections với tracks
        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        # Cập nhật matched tracks
        for track_idx, det_idx in matched:
            bbox, feature = detections[det_idx]
            self.tracks[track_idx].update(bbox, feature)

        # Tạo tracks mới cho unmatched detections
        for det_idx in unmatched_dets:
            bbox, feature = detections[det_idx]
            self.tracks.append(Track(self.next_id, bbox, feature))
            self.next_id += 1

        # Đánh dấu unmatched tracks là missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Xóa tracks đã chết
        self.tracks = [t for t in self.tracks
                      if t.time_since_update <= self.max_age]

        # Trả về confirmed tracks
        return [t for t in self.tracks if t.state == 'confirmed']

    def _match(self, detections):
        """Khớp detections với tracks"""

        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Tính toán IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for i, track in enumerate(self.tracks):
            for j, (det_bbox, _) in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det_bbox)

        # Hungarian matching (đơn giản hóa: greedy)
        matched_indices = []

        for _ in range(min(len(self.tracks), len(detections))):
            max_iou = iou_matrix.max()
            if max_iou < 0.3:  # Ngưỡng
                break

            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched_indices.append((i, j))

            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        matched_tracks = [i for i, _ in matched_indices]
        matched_dets = [j for _, j in matched_indices]

        unmatched_tracks = [i for i in range(len(self.tracks))
                           if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections))
                         if j not in matched_dets]

        return matched_indices, unmatched_dets, unmatched_tracks

    def _iou(self, bbox1, bbox2):
        """Tính toán IoU giữa hai bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        return iou
```

---

## 6. So Sánh Hiệu Năng

| Tracker | Tốc Độ | Độ Chính Xác | Độ Ổn Định | Trường Hợp Sử Dụng |
|---------|-------|----------|------------|----------|
| Centroid | Rất Nhanh | Thấp | Thấp | Cảnh đơn giản |
| MOSSE | Nhanh | Trung Bình | Trung Bình | Thời gian thực |
| KCF | Nhanh | Tốt | Tốt | Chung |
| CSRT | Chậm | Xuất Sắc | Xuất Sắc | Quan trọng độ chính xác |
| Kalman | Nhanh | Tốt | Tốt | Chuyển động mượt |
| DeepSORT | Trung Bình | Xuất Sắc | Xuất Sắc | Nhiều đối tượng |

---

## 7. Best Practices (Thực Hành Tốt)

### A. Lựa Chọn Tracker

```python
def select_tracker(scenario):
    """Chọn tracker phù hợp cho kịch bản"""

    if scenario == 'fast_cpu':
        return cv2.TrackerMOSSE_create()

    elif scenario == 'accuracy':
        return cv2.TrackerCSRT_create()

    elif scenario == 'balanced':
        return cv2.TrackerKCF_create()

    elif scenario == 'multi_object':
        return CentroidTracker()

    else:
        return cv2.TrackerKCF_create()
```

### B. Re-detection (Phát Hiện Lại)

```python
class RobustTracker:
    """Tracker với khả năng re-detection"""

    def __init__(self, detector, tracker_type='KCF'):
        self.detector = detector
        self.tracker = None
        self.tracker_type = tracker_type
        self.bbox = None
        self.failures = 0
        self.max_failures = 5

    def update(self, frame):
        """Cập nhật tracker hoặc re-detect"""

        if self.tracker is None or self.failures >= self.max_failures:
            # Re-detect
            detections = self.detector.detect(frame)

            if len(detections) > 0:
                self.bbox = detections[0]
                self._reinit_tracker(frame, self.bbox)
                self.failures = 0
                return True, self.bbox
            else:
                return False, None

        # Track
        success, bbox = self.tracker.update(frame)

        if success:
            self.bbox = bbox
            self.failures = 0
            return True, bbox
        else:
            self.failures += 1
            return False, None

    def _reinit_tracker(self, frame, bbox):
        """Khởi tạo lại tracker"""
        if self.tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == 'CSRT':
            self.tracker = cv2.TrackerCSRT_create()
        else:
            self.tracker = cv2.TrackerMOSSE_create()

        self.tracker.init(frame, bbox)
```

---

**Ngày tạo**: Tháng 1/2025
