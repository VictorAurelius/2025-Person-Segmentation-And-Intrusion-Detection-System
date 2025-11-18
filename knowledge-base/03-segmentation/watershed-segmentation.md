# Watershed Segmentation

## 1. Khái Niệm

**Watershed Segmentation** là thuật toán dựa trên topological interpretation của image, xử lý grayscale image như một topographic surface.

### A. Metaphor

Tưởng tượng grayscale image như địa hình:
- **Bright regions** = Peaks (đỉnh núi)
- **Dark regions** = Valleys (thung lũng)
- **Gradients** = Slopes (dốc)

Nếu đổ nước vào thung lũng, nước sẽ dâng lên và tạo thành các **watersheds** (phân thủy).

---

## 2. Algorithm Principle

### A. Flooding Process

```
1. Tìm local minima (valleys)
2. Đánh dấu mỗi valley với unique label
3. "Đổ nước" vào valleys (flooding)
4. Khi 2 "vùng nước" gặp nhau → Watershed line
5. Watersheds tạo thành boundaries giữa objects
```

### B. Visualization

```
Image Intensity Profile:
      ╱╲        ╱╲
     ╱  ╲      ╱  ╲
    ╱    ╲____╱    ╲
   ╱                ╲
  ╱                  ╲

→ Valleys (markers) at ____
→ Water rises ↑
→ Watershed at peaks ╱╲
```

---

## 3. Basic Implementation

### A. Simple Watershed

```python
import cv2
import numpy as np
from scipy import ndimage

# Read image
image = cv2.imread('coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background (dilation)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Sure foreground (distance transform)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Label markers
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels (background = 1, not 0)
markers = markers + 1

# Mark unknown region as 0
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(image, markers)

# Mark boundaries
image[markers == -1] = [255, 0, 0]  # Red boundaries

cv2.imshow('Result', image)
cv2.waitKey(0)
```

---

## 4. Marker-Controlled Watershed

### A. Why Markers?

**Problem:** Basic watershed over-segments (too many regions)

**Solution:** Use markers to guide segmentation

### B. Implementation

```python
class WatershedSegmentation:
    """Marker-controlled watershed segmentation"""

    def __init__(self):
        self.markers = None

    def segment(self, image, min_distance=20):
        """Segment image using watershed"""

        # Prepare image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        # Sure foreground
        _, sure_fg = cv2.threshold(
            dist_transform,
            0.5 * dist_transform.max(),
            255,
            0
        )
        sure_fg = np.uint8(sure_fg)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        _, self.markers = cv2.connectedComponents(sure_fg)

        # Add 1 to markers (background != 0)
        self.markers = self.markers + 1

        # Mark unknown as 0
        self.markers[unknown == 255] = 0

        # Apply watershed
        self.markers = cv2.watershed(image, self.markers)

        return self.markers

    def visualize(self, image, markers=None):
        """Visualize segmentation result"""

        if markers is None:
            markers = self.markers

        if markers is None:
            raise ValueError("No markers available. Run segment() first.")

        result = image.copy()

        # Color each segment
        num_segments = markers.max()

        for i in range(2, num_segments + 1):
            mask = (markers == i).astype(np.uint8)

            # Random color
            color = np.random.randint(0, 255, 3).tolist()

            # Apply color
            result[mask == 1] = color

        # Mark boundaries
        result[markers == -1] = [255, 0, 0]

        return result

    def get_contours(self, markers=None):
        """Extract contours from markers"""

        if markers is None:
            markers = self.markers

        contours = []
        num_segments = markers.max()

        for i in range(2, num_segments + 1):
            mask = (markers == i).astype(np.uint8) * 255

            # Find contour
            cnts, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(cnts) > 0:
                contours.append(cnts[0])

        return contours


# Usage
segmenter = WatershedSegmentation()

# Segment
markers = segmenter.segment(image)

# Visualize
result = segmenter.visualize(image)
cv2.imshow('Watershed Segmentation', result)

# Get contours
contours = segmenter.get_contours()
print(f"Found {len(contours)} objects")
```

---

## 5. Distance Transform

### A. Concept

Distance Transform tính khoảng cách từ mỗi foreground pixel đến nearest background pixel.

```python
# Binary image
binary = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=np.uint8)

# Distance transform
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Result (approximate):
# [0, 0, 0, 0, 0]
# [0, 1, 1, 1, 0]
# [0, 1, 2, 1, 0]  ← Center has highest distance
# [0, 1, 1, 1, 0]
# [0, 0, 0, 0, 0]
```

### B. Distance Metrics

```python
# Euclidean (L2)
dist_l2 = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Manhattan (L1)
dist_l1 = cv2.distanceTransform(binary, cv2.DIST_L1, 3)

# Chessboard (L∞)
dist_linf = cv2.distanceTransform(binary, cv2.DIST_C, 3)
```

### C. Peak Detection

```python
def find_peaks(dist_transform, min_distance=20):
    """Find peaks in distance transform"""

    from scipy.ndimage import maximum_filter

    # Local maxima
    local_max = maximum_filter(dist_transform, size=min_distance) == dist_transform

    # Threshold (only significant peaks)
    threshold = 0.5 * dist_transform.max()
    peaks = local_max & (dist_transform > threshold)

    # Get coordinates
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

## 8. Separating Touching Objects

### A. Problem

Touching/overlapping objects appear as single blob.

### B. Solution

```python
def separate_touching_objects(binary_mask):
    """Separate touching objects using watershed"""

    # Distance transform
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # Find peaks (object centers)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Dilate to find background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    # Need 3-channel image
    image_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_3ch, markers)

    # Create separated mask
    separated = np.zeros_like(binary_mask)
    separated[markers > 1] = 255

    return separated, markers
```

---

## 9. Parameter Tuning

### A. Distance Transform Threshold

```python
# Low threshold (0.3-0.5): More aggressive separation
_, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

# Medium threshold (0.5-0.7): Balanced (recommended)
_, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

# High threshold (0.7-0.9): Conservative separation
_, sure_fg = cv2.threshold(dist_transform, 0.8 * dist_transform.max(), 255, 0)
```

### B. Morphological Operations

```python
# Weak opening (less noise removal)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# Strong opening (more noise removal)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)

# Small dilation (tight background)
sure_bg = cv2.dilate(opening, kernel, iterations=1)

# Large dilation (wide background)
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

## 11. Post-Processing

### A. Merge Small Regions

```python
def merge_small_regions(markers, min_size=100):
    """Merge regions smaller than min_size"""

    unique_markers = np.unique(markers)
    unique_markers = unique_markers[unique_markers > 1]  # Exclude 0, 1, -1

    for marker_id in unique_markers:
        region_mask = (markers == marker_id)
        region_size = np.sum(region_mask)

        if region_size < min_size:
            # Find neighboring regions
            dilated = cv2.dilate(region_mask.astype(np.uint8), np.ones((3, 3)))
            neighbors = markers[dilated == 1]
            neighbors = neighbors[neighbors != marker_id]
            neighbors = neighbors[neighbors > 1]

            if len(neighbors) > 0:
                # Merge with most common neighbor
                merge_to = np.bincount(neighbors).argmax()
                markers[region_mask] = merge_to

    return markers
```

### B. Smooth Boundaries

```python
def smooth_watershed_boundaries(markers):
    """Smooth watershed boundaries"""

    smoothed = markers.copy()

    # For each region
    unique_markers = np.unique(markers)
    unique_markers = unique_markers[unique_markers > 1]

    for marker_id in unique_markers:
        # Get region mask
        mask = (markers == marker_id).astype(np.uint8) * 255

        # Find contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Smooth contour
            epsilon = 0.01 * cv2.arcLength(contours[0], True)
            smoothed_contour = cv2.approxPolyDP(contours[0], epsilon, True)

            # Update markers
            mask_new = np.zeros_like(mask)
            cv2.drawContours(mask_new, [smoothed_contour], 0, 255, -1)

            smoothed[mask_new == 255] = marker_id

    return smoothed
```

---

## 12. Complete Example

```python
class AdvancedWatershed:
    """Advanced watershed segmentation pipeline"""

    def __init__(self, min_area=100, dist_threshold=0.6):
        self.min_area = min_area
        self.dist_threshold = dist_threshold

    def segment(self, image):
        """Complete segmentation pipeline"""

        # 1. Preprocessing
        processed = self._preprocess(image)

        # 2. Find markers
        markers = self._find_markers(processed)

        # 3. Apply watershed
        markers = cv2.watershed(image, markers)

        # 4. Post-processing
        markers = self._postprocess(markers)

        return markers

    def _preprocess(self, image):
        """Preprocess image"""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Threshold
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphology
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        return binary

    def _find_markers(self, binary):
        """Find watershed markers"""

        # Distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Sure foreground
        _, sure_fg = cv2.threshold(dist, self.dist_threshold * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Sure background
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)

        # Unknown
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        return markers

    def _postprocess(self, markers):
        """Post-process markers"""

        # Merge small regions
        markers = merge_small_regions(markers, min_size=self.min_area)

        return markers


# Usage
segmenter = AdvancedWatershed(min_area=200, dist_threshold=0.6)

markers = segmenter.segment(image)

# Visualize
result = image.copy()
result[markers == -1] = [255, 0, 0]

cv2.imshow('Watershed Result', result)
cv2.waitKey(0)
```

---

**Ngày tạo**: Tháng 1/2025
