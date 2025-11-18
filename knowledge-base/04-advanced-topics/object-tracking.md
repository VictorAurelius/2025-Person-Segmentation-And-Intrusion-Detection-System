# Object Tracking

## 1. Khái Niệm

**Object Tracking** là quá trình locating một moving object (hoặc nhiều objects) theo thời gian sử dụng camera.

### A. Tracking vs Detection

| Aspect | Detection | Tracking |
|--------|-----------|----------|
| Goal | Find objects in single frame | Follow objects across frames |
| Speed | Slower | Faster |
| Consistency | No history | Uses temporal information |
| ID | No persistent ID | Maintains object ID |

---

## 2. Centroid Tracking

### A. Algorithm

```
1. Detect objects in frame
2. Calculate centroids
3. Match with previous centroids
4. Update IDs
```

### B. Implementation

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
        """Register new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Deregister object"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, bboxes):
        """Update tracker with new detections"""

        # No detections
        if len(bboxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # Deregister if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        # Calculate centroids from bboxes
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")

        for (i, (x, y, w, h)) in enumerate(bboxes):
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)

        # No existing objects
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        # Match existing objects to new centroids
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distance matrix
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find minimum distance matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Update object centroid
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle disappeared objects
            unused_rows = set(range(0, D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new objects
            unused_cols = set(range(0, D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# Usage
tracker = CentroidTracker(max_disappeared=50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects (returns bounding boxes)
    bboxes = detect_objects(frame)  # [(x, y, w, h), ...]

    # Update tracker
    objects = tracker.update(bboxes)

    # Draw tracked objects
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

### A. Available Trackers

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

### B. Basic Usage

```python
# Select ROI
bbox = cv2.selectROI('Frame', frame, False)

# Create tracker
tracker = cv2.TrackerCSRT_create()

# Initialize
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    success, bbox = tracker.update(frame)

    if success:
        # Draw bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### C. Multi-Object Tracking

```python
class MultiTracker:
    """Track multiple objects using OpenCV trackers"""

    def __init__(self, tracker_type='CSRT'):
        self.tracker_type = tracker_type
        self.trackers = []
        self.colors = []

    def add(self, frame, bbox):
        """Add new tracker"""

        # Create tracker
        if self.tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        else:
            tracker = cv2.TrackerCSRT_create()

        # Initialize
        tracker.init(frame, bbox)

        # Store
        self.trackers.append(tracker)
        self.colors.append(tuple(np.random.randint(0, 255, 3).tolist()))

    def update(self, frame):
        """Update all trackers"""

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
        """Remove tracker"""
        if 0 <= tracker_id < len(self.trackers):
            del self.trackers[tracker_id]
            del self.colors[tracker_id]


# Usage
multi_tracker = MultiTracker(tracker_type='CSRT')

# Select multiple ROIs
while True:
    bbox = cv2.selectROI('Select Object', frame, False)
    if bbox[2] == 0 or bbox[3] == 0:
        break

    multi_tracker.add(frame, bbox)

# Tracking loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update all trackers
    results = multi_tracker.update(frame)

    # Draw results
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

### A. Concept

Kalman Filter predicts object position and corrects prediction using measurements.

### B. Implementation

```python
class KalmanTracker:
    """Kalman filter-based object tracker"""

    def __init__(self):
        # Create Kalman filter
        # State: [x, y, dx, dy] (position + velocity)
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
        """Predict next position"""
        prediction = self.kalman.predict()
        return prediction[:2].flatten()  # Return [x, y]

    def correct(self, measurement):
        """Correct prediction with measurement"""
        measurement = np.array([[np.float32(measurement[0])],
                               [np.float32(measurement[1])]])
        corrected = self.kalman.correct(measurement)
        return corrected[:2].flatten()


# Usage
tracker = KalmanTracker()

# Initialize with first detection
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

    # Predict
    predicted = tracker.predict()

    # Detect object (returns centroid or None)
    detection = detect_object(frame)

    if detection is not None:
        # Correct with measurement
        corrected = tracker.correct(detection)

        # Draw detection (red) and corrected (green)
        cv2.circle(frame, tuple(detection.astype(int)), 5, (0, 0, 255), -1)
        cv2.circle(frame, tuple(corrected.astype(int)), 5, (0, 255, 0), -1)
    else:
        # No detection, use prediction only
        cv2.circle(frame, tuple(predicted.astype(int)), 5, (255, 0, 0), -1)

    cv2.imshow('Kalman Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 5. DeepSORT-inspired Tracking

### A. Simplified DeepSORT

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
        self.state = 'tentative'  # tentative or confirmed

    def update(self, bbox, feature=None):
        """Update track with new detection"""
        self.bbox = bbox
        if feature is not None:
            self.feature = feature
        self.age += 1
        self.time_since_update = 0
        self.hit_streak += 1

        if self.hit_streak >= 3:
            self.state = 'confirmed'

    def mark_missed(self):
        """Mark track as missed"""
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
        Update tracks with new detections

        detections: list of (bbox, feature)
        """

        # Predict (in full DeepSORT, use Kalman filter here)

        # Match detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._match(detections)

        # Update matched tracks
        for track_idx, det_idx in matched:
            bbox, feature = detections[det_idx]
            self.tracks[track_idx].update(bbox, feature)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox, feature = detections[det_idx]
            self.tracks.append(Track(self.next_id, bbox, feature))
            self.next_id += 1

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Remove dead tracks
        self.tracks = [t for t in self.tracks
                      if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        return [t for t in self.tracks if t.state == 'confirmed']

    def _match(self, detections):
        """Match detections to tracks"""

        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))

        for i, track in enumerate(self.tracks):
            for j, (det_bbox, _) in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det_bbox)

        # Hungarian matching (simplified: greedy)
        matched_indices = []

        for _ in range(min(len(self.tracks), len(detections))):
            max_iou = iou_matrix.max()
            if max_iou < 0.3:  # Threshold
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
        """Calculate IoU between two bounding boxes"""
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

## 6. Performance Comparison

| Tracker | Speed | Accuracy | Robustness | Use Case |
|---------|-------|----------|------------|----------|
| Centroid | Very Fast ✅ | Low | Low | Simple scenes |
| MOSSE | Fast ✅ | Medium | Medium | Real-time |
| KCF | Fast ✅ | Good | Good | General |
| CSRT | Slow ⚠️ | Excellent ✅ | Excellent ✅ | Accuracy critical |
| Kalman | Fast ✅ | Good | Good | Smooth motion |
| DeepSORT | Medium | Excellent ✅ | Excellent ✅ | Multi-object |

---

## 7. Best Practices

### A. Tracker Selection

```python
def select_tracker(scenario):
    """Select appropriate tracker for scenario"""

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

### B. Re-detection

```python
class RobustTracker:
    """Tracker with re-detection capability"""

    def __init__(self, detector, tracker_type='KCF'):
        self.detector = detector
        self.tracker = None
        self.tracker_type = tracker_type
        self.bbox = None
        self.failures = 0
        self.max_failures = 5

    def update(self, frame):
        """Update tracker or re-detect"""

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
        """Re-initialize tracker"""
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
