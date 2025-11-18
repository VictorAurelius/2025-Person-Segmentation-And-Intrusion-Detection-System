# OpenCV Essentials

## 1. OpenCV Introduction

### A. Giới Thiệu

**OpenCV** (Open Source Computer Vision Library) là thư viện mã nguồn mở cho computer vision và machine learning.

**Tính năng chính:**
- 2500+ optimized algorithms
- Hỗ trợ Python, C++, Java
- Cross-platform (Windows, Linux, macOS, Android, iOS)
- Real-time processing
- GPU acceleration (CUDA)

### B. Installation

```bash
# Basic OpenCV
pip install opencv-python

# With contrib modules (extra features)
pip install opencv-contrib-python

# Verify installation
python -c "import cv2; print(cv2.__version__)"
```

### C. Import Convention

```python
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
```

---

## 2. Video Processing

### A. Video Capture

#### From File

```python
# Open video file
cap = cv2.VideoCapture('video.mp4')

# Check if opened
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Total frames: {frame_count}")
```

#### From Webcam

```python
# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set FPS
cap.set(cv2.CAP_PROP_FPS, 30)
```

### B. Reading Frames

```python
while True:
    # Read frame
    ret, frame = cap.read()

    # Check if frame read successfully
    if not ret:
        print("End of video or error")
        break

    # Process frame here
    cv2.imshow('Frame', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

### C. Video Writing

```python
# Define codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG'

# Create VideoWriter
out = cv2.VideoWriter(
    'output.mp4',
    fourcc,
    30.0,  # FPS
    (640, 480)  # Resolution (width, height)
)

# Write frames
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Process frame
        processed = process_frame(frame)

        # Write
        out.write(processed)
    else:
        break

# Release
out.release()
```

### D. Advanced Video Operations

```python
# Seek to specific frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Jump to frame 100

# Get current frame position
current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

# Get video duration
duration = frame_count / fps
print(f"Duration: {duration:.2f} seconds")

# Skip frames (for performance)
skip_frames = 2
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue  # Skip this frame

    # Process every Nth frame
    process_frame(frame)
```

---

## 3. Background Subtraction

### A. MOG2

```python
# Create MOG2 background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,           # Number of frames to learn from
    varThreshold=16,       # Threshold for detection
    detectShadows=True     # Detect and mark shadows
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Display
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
```

**Parameters:**
- `history`: Larger = more stable, slower adaptation
- `varThreshold`: Lower = more sensitive
- `detectShadows`: True = marks shadows (value 127)

### B. KNN

```python
# Create KNN background subtractor
bg_subtractor = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)

# Usage same as MOG2
fg_mask = bg_subtractor.apply(frame)
```

**KNN vs MOG2:**
- KNN: Better with noise, slower
- MOG2: Faster, good for most cases

### C. Background Image

```python
# Get learned background
background = bg_subtractor.getBackgroundImage()

# Display
cv2.imshow('Learned Background', background)
```

---

## 4. Contour Detection

### A. Find Contours

```python
# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,  # Retrieval mode
    cv2.CHAIN_APPROX_SIMPLE  # Approximation method
)

print(f"Found {len(contours)} contours")
```

**Retrieval Modes:**
- `RETR_EXTERNAL`: Only outermost contours
- `RETR_LIST`: All contours, no hierarchy
- `RETR_TREE`: Full hierarchy

**Approximation Methods:**
- `CHAIN_APPROX_NONE`: All points
- `CHAIN_APPROX_SIMPLE`: Only corner points (saves memory)

### B. Draw Contours

```python
# Draw all contours
cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

# Draw specific contour
cv2.drawContours(frame, contours, 0, (0, 255, 0), 2)  # Draw first contour

# Draw filled
cv2.drawContours(frame, contours, -1, (0, 255, 0), -1)
```

### C. Contour Properties

```python
for contour in contours:
    # Area
    area = cv2.contourArea(contour)

    # Perimeter
    perimeter = cv2.arcLength(contour, closed=True)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Minimum area rectangle (rotated)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

    # Centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Convex hull
    hull = cv2.convexHull(contour)

    # Aspect ratio
    aspect_ratio = float(w) / h

    # Extent (ratio of contour area to bounding box area)
    rect_area = w * h
    extent = float(area) / rect_area

    # Solidity (ratio of contour area to convex hull area)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
```

### D. Contour Filtering

```python
# Filter by area
min_area = 500
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# Filter by aspect ratio (for vertical objects like people)
def filter_by_aspect_ratio(contours, min_ratio=0.3, max_ratio=3.0):
    filtered = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        if min_ratio <= aspect_ratio <= max_ratio:
            filtered.append(c)
    return filtered

# Filter by solidity (compact objects)
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

## 5. ROI Operations

### A. Point in Polygon

```python
# Define ROI polygon
roi_points = np.array([[100, 100], [300, 100], [300, 300], [100, 300]])

# Check if point is inside
point = (200, 200)
result = cv2.pointPolygonTest(roi_points, point, False)

if result >= 0:
    print("Point inside ROI")
else:
    print("Point outside ROI")
```

### B. Mask Creation

```python
# Create mask from polygon
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [roi_points], 255)

# Apply mask to image
masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
```

### C. ROI Overlap

```python
def calculate_overlap(contour, roi_polygon):
    """Calculate overlap percentage between contour and ROI"""
    # Create masks
    h, w = 1080, 1920  # Frame size
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    # Fill masks
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    cv2.fillPoly(roi_mask, [roi_polygon], 255)

    # Calculate intersection
    intersection = cv2.bitwise_and(contour_mask, roi_mask)
    intersection_area = cv2.countNonZero(intersection)

    # Calculate contour area
    contour_area = cv2.contourArea(contour)

    # Overlap percentage
    if contour_area > 0:
        overlap = intersection_area / contour_area
    else:
        overlap = 0

    return overlap
```

---

## 6. Performance Optimization

### A. Reduce Resolution

```python
# Resize for processing
scale = 0.5
small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

# Process smaller frame
processed_small = process(small_frame)

# Resize back if needed
processed = cv2.resize(processed_small, (frame.shape[1], frame.shape[0]))
```

### B. ROI Processing

```python
# Only process ROI area
x, y, w, h = 100, 100, 400, 300
roi = frame[y:y+h, x:x+w]

# Process ROI
processed_roi = process(roi)

# Put back
frame[y:y+h, x:x+w] = processed_roi
```

### C. Frame Skipping

```python
frame_counter = 0
process_every_n_frames = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % process_every_n_frames == 0:
        # Process this frame
        processed = process(frame)
    else:
        # Use previous result
        pass

    cv2.imshow('Frame', frame)
```

### D. Use Grayscale

```python
# Convert to grayscale early
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Process grayscale (3x faster than color)
processed = process_grayscale(gray)
```

---

## 7. Useful Utilities

### A. FPS Counter

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

# Usage
fps_counter = FPSCounter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    fps_counter.update()

    # Display FPS
    fps = fps_counter.get_fps()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
```

### B. Progress Bar

```python
def process_video_with_progress(video_path):
    """Process video with progress display"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed = process(frame)

        # Update progress
        frame_number += 1
        progress = (frame_number / total_frames) * 100

        print(f"\rProgress: {progress:.1f}% ({frame_number}/{total_frames})", end='')

    print("\nDone!")
    cap.release()
```

### C. Auto-restart on Error

```python
def robust_capture(source):
    """Robust video capture with auto-restart"""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Failed to open. Retry {retry_count + 1}/{max_retries}")
            retry_count += 1
            time.sleep(1)
            continue

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Lost connection. Reconnecting...")
                break

            # Process frame
            yield frame

        cap.release()
        retry_count += 1

    print("Max retries reached. Exiting.")
```

---

## 8. Mouse Callbacks

### A. Basic Callback

```python
def mouse_callback(event, x, y, flags, param):
    """Handle mouse events"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left click at ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Right click at ({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to ({x}, {y})")

# Set callback
cv2.namedWindow('Window')
cv2.setMouseCallback('Window', mouse_callback)

while True:
    cv2.imshow('Window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### B. Interactive ROI Selection

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
            # Finish ROI
            if len(self.points) >= 3:
                print(f"ROI completed with {len(self.points)} points")
                self.drawing = False

    def draw_roi(self, frame):
        """Draw current ROI on frame"""
        if len(self.points) > 0:
            # Draw points
            for point in self.points:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

            # Draw lines
            if len(self.points) > 1:
                pts = np.array(self.points, np.int32)
                cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

        return frame

# Usage
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

## 9. Trackbars

```python
def nothing(x):
    """Dummy callback for trackbars"""
    pass

# Create window with trackbars
cv2.namedWindow('Controls')

# Add trackbars
cv2.createTrackbar('Threshold', 'Controls', 127, 255, nothing)
cv2.createTrackbar('Blur', 'Controls', 5, 50, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get trackbar values
    threshold_value = cv2.getTrackbarPos('Threshold', 'Controls')
    blur_value = cv2.getTrackbarPos('Blur', 'Controls')

    # Make blur value odd
    if blur_value % 2 == 0:
        blur_value += 1

    # Apply processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    cv2.imshow('Result', binary)
    cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 10. Common Patterns

### A. Video Processing Template

```python
def process_video(input_path, output_path=None):
    """Standard video processing template"""
    # Open video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error opening video")
        return

    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create writer if saving
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process loop
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # **YOUR PROCESSING HERE**
            processed_frame = your_processing_function(frame)

            # Save if writer exists
            if out:
                out.write(processed_frame)

            # Display
            cv2.imshow('Processing', processed_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        print(f"Processed {frame_count} frames")
```

### B. Multi-window Display

```python
def show_multiple_views(frame, fg_mask, edges):
    """Display multiple views in organized layout"""
    # Resize for display
    h, w = 300, 400

    frame_resized = cv2.resize(frame, (w, h))
    mask_resized = cv2.resize(fg_mask, (w, h))
    edges_resized = cv2.resize(edges, (w, h))

    # Convert grayscale to BGR for stacking
    mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)

    # Stack horizontally
    top_row = np.hstack([frame_resized, mask_bgr])
    bottom_row = np.hstack([edges_bgr, np.zeros((h, w, 3), dtype=np.uint8)])

    # Stack vertically
    combined = np.vstack([top_row, bottom_row])

    # Add labels
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
