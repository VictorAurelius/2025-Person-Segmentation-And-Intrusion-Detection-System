# Contour Analysis

## 1. Khái Niệm

**Contour** là curve nối các điểm liên tục có cùng color hoặc intensity. Trong context của OpenCV, contours là đường biên của objects.

### A. Definition

```python
# Contour: List of points defining boundary
contour = np.array([
    [[100, 100]],
    [[150, 100]],
    [[150, 150]],
    [[100, 150]]
])  # Shape: (N, 1, 2)
```

---

## 2. Finding Contours

### A. Basic Usage

```python
import cv2
import numpy as np

# Read and prepare image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to get binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,        # Retrieval mode
    cv2.CHAIN_APPROX_SIMPLE   # Approximation method
)

print(f"Found {len(contours)} contours")
```

### B. Retrieval Modes

```python
# RETR_EXTERNAL: Only outermost contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# RETR_LIST: All contours, no hierarchy
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# RETR_TREE: All contours with full hierarchy
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# RETR_CCOMP: Two-level hierarchy
contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
```

**Hierarchy Format:**
```
hierarchy[i] = [next, previous, first_child, parent]
```

### C. Approximation Methods

```python
# CHAIN_APPROX_NONE: Store all points
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"Points: {len(contours[0])}")  # Many points

# CHAIN_APPROX_SIMPLE: Store only corner points (recommended)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Points: {len(contours[0])}")  # Fewer points
```

---

## 3. Drawing Contours

### A. Basic Drawing

```python
# Draw all contours
result = image.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Draw specific contour
cv2.drawContours(result, contours, 0, (0, 255, 0), 2)  # First contour

# Draw filled
cv2.drawContours(result, contours, -1, (0, 255, 0), -1)  # thickness=-1
```

### B. Custom Drawing

```python
def draw_contours_custom(image, contours):
    """Draw contours with different colors"""
    result = image.copy()

    for i, contour in enumerate(contours):
        # Random color for each contour
        color = np.random.randint(0, 255, 3).tolist()

        # Draw contour
        cv2.drawContours(result, [contour], -1, color, 2)

        # Draw contour index
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(result, str(i), (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return result
```

---

## 4. Contour Properties

### A. Area

```python
for contour in contours:
    area = cv2.contourArea(contour)
    print(f"Area: {area} pixels²")
```

### B. Perimeter

```python
for contour in contours:
    # Closed contour
    perimeter = cv2.arcLength(contour, closed=True)
    print(f"Perimeter: {perimeter} pixels")

    # Open contour
    perimeter_open = cv2.arcLength(contour, closed=False)
```

### C. Bounding Rectangle

```python
for contour in contours:
    # Straight bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"BBox: ({x}, {y}, {w}, {h})")
```

### D. Rotated Rectangle

```python
for contour in contours:
    # Minimum area rectangle (can be rotated)
    rect = cv2.minAreaRect(contour)
    # rect = ((center_x, center_y), (width, height), angle)

    # Get box points
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
```

### E. Minimum Enclosing Circle

```python
for contour in contours:
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    cv2.circle(image, center, radius, (255, 0, 0), 2)
```

### F. Centroid (Center of Mass)

```python
for contour in contours:
    M = cv2.moments(contour)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
        print(f"Centroid: ({cx}, {cy})")
```

### G. Convex Hull

```python
for contour in contours:
    hull = cv2.convexHull(contour)

    # Draw hull
    cv2.drawContours(image, [hull], 0, (0, 255, 255), 2)
```

---

## 5. Contour Approximation

### A. Douglas-Peucker Algorithm

```python
for contour in contours:
    # Approximate contour with fewer points
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    print(f"Original points: {len(contour)}")
    print(f"Approximated points: {len(approx)}")

    # Draw approximation
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
```

**Epsilon Parameter:**
```python
# Small epsilon: More detailed (more points)
epsilon = 0.001 * cv2.arcLength(contour, True)

# Medium epsilon: Balanced (recommended)
epsilon = 0.01 * cv2.arcLength(contour, True)

# Large epsilon: Simple (fewer points)
epsilon = 0.1 * cv2.arcLength(contour, True)
```

### B. Shape Detection

```python
def detect_shape(contour):
    """Detect shape from contour"""

    # Approximate
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Number of vertices
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # Check if square or rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices > 5:
        # Check if circle
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius ** 2

        if abs(area - circle_area) / circle_area < 0.2:
            return "Circle"
        else:
            return "Ellipse"
    else:
        return "Unknown"
```

---

## 6. Contour Features

### A. Aspect Ratio

```python
x, y, w, h = cv2.boundingRect(contour)
aspect_ratio = float(w) / h

print(f"Aspect Ratio: {aspect_ratio}")

# Classification
if aspect_ratio > 2:
    print("Horizontal object")
elif aspect_ratio < 0.5:
    print("Vertical object")
else:
    print("Balanced object")
```

### B. Extent

**Extent** = Contour Area / Bounding Rectangle Area

```python
area = cv2.contourArea(contour)
x, y, w, h = cv2.boundingRect(contour)
rect_area = w * h
extent = float(area) / rect_area

print(f"Extent: {extent:.2f}")

# Interpretation:
# extent ≈ 1.0: Fills bounding box (rectangle)
# extent ≈ 0.79: Circle (π/4)
# extent < 0.5: Irregular shape
```

### C. Solidity

**Solidity** = Contour Area / Convex Hull Area

```python
area = cv2.contourArea(contour)
hull = cv2.convexHull(contour)
hull_area = cv2.contourArea(hull)
solidity = float(area) / hull_area if hull_area > 0 else 0

print(f"Solidity: {solidity:.2f}")

# Interpretation:
# solidity ≈ 1.0: Convex shape
# solidity < 0.8: Has concavities
```

### D. Equivalent Diameter

```python
area = cv2.contourArea(contour)
equi_diameter = np.sqrt(4 * area / np.pi)

print(f"Equivalent Diameter: {equi_diameter:.2f} pixels")
```

### E. Orientation

```python
# Fit ellipse (requires at least 5 points)
if len(contour) >= 5:
    ellipse = cv2.fitEllipse(contour)
    (x, y), (MA, ma), angle = ellipse

    print(f"Orientation: {angle:.2f} degrees")

    # Draw ellipse
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
```

---

## 7. Contour Filtering

### A. By Area

```python
def filter_by_area(contours, min_area=500, max_area=50000):
    """Filter contours by area"""
    filtered = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered.append(contour)

    return filtered

# Usage
large_contours = filter_by_area(contours, min_area=1000)
```

### B. By Shape

```python
def filter_by_aspect_ratio(contours, min_ratio=0.5, max_ratio=2.0):
    """Filter contours by aspect ratio"""
    filtered = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        if min_ratio <= aspect_ratio <= max_ratio:
            filtered.append(contour)

    return filtered

# Example: Filter vertical objects (people, poles)
vertical_contours = filter_by_aspect_ratio(contours, min_ratio=0.3, max_ratio=0.8)
```

### C. By Solidity

```python
def filter_by_solidity(contours, min_solidity=0.8):
    """Filter out contours with holes/concavities"""
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

# Filter compact shapes only
compact_contours = filter_by_solidity(contours, min_solidity=0.9)
```

### D. Combined Filtering

```python
class ContourFilter:
    """Filter contours by multiple criteria"""

    def __init__(self, min_area=500, max_area=50000,
                 min_aspect=0.3, max_aspect=3.0,
                 min_solidity=0.7):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_solidity = min_solidity

    def filter(self, contours):
        """Apply all filters"""
        filtered = []

        for contour in contours:
            if self._meets_criteria(contour):
                filtered.append(contour)

        return filtered

    def _meets_criteria(self, contour):
        """Check if contour meets all criteria"""

        # Area
        area = cv2.contourArea(contour)
        if not (self.min_area <= area <= self.max_area):
            return False

        # Aspect ratio
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


# Usage
filter = ContourFilter(min_area=1000, max_area=10000,
                        min_aspect=0.5, max_aspect=2.0,
                        min_solidity=0.8)

filtered_contours = filter.filter(contours)
```

---

## 8. Contour Matching

### A. Shape Matching

```python
# Template contour
template = contours[0]

# Compare with other contours
for i, contour in enumerate(contours[1:], 1):
    match = cv2.matchShapes(template, contour, cv2.CONTOURS_MATCH_I1, 0)

    print(f"Contour {i}: match = {match:.4f}")

    # Low value = good match
    if match < 0.1:
        print(f"  → Similar to template!")
```

**Match Methods:**
- `CONTOURS_MATCH_I1`: Hu moments method 1
- `CONTOURS_MATCH_I2`: Hu moments method 2
- `CONTOURS_MATCH_I3`: Hu moments method 3

---

## 9. Hierarchy Analysis

### A. Parent-Child Relationship

```python
def analyze_hierarchy(contours, hierarchy):
    """Analyze contour hierarchy"""

    # hierarchy[i] = [next, previous, first_child, parent]

    for i, contour in enumerate(contours):
        h = hierarchy[0][i]
        next_idx, prev_idx, child_idx, parent_idx = h

        area = cv2.contourArea(contour)

        print(f"Contour {i}:")
        print(f"  Area: {area}")
        print(f"  Parent: {parent_idx}")
        print(f"  First Child: {child_idx}")

        if parent_idx == -1:
            print(f"  → Outer contour")
        else:
            print(f"  → Inner contour (hole)")
```

### B. Detect Objects with Holes

```python
def find_objects_with_holes(contours, hierarchy):
    """Find contours that contain holes"""

    objects_with_holes = []

    for i, contour in enumerate(contours):
        h = hierarchy[0][i]
        child_idx = h[2]

        # Has child = has hole
        if child_idx != -1:
            objects_with_holes.append((i, contour))

    return objects_with_holes
```

---

## 10. Advanced Analysis

### A. Contour Moments

```python
def analyze_moments(contour):
    """Analyze contour using moments"""

    M = cv2.moments(contour)

    # Central moments
    mu20 = M['mu20']
    mu02 = M['mu02']
    mu11 = M['mu11']

    # Orientation
    if (mu20 - mu02) != 0:
        theta = 0.5 * np.arctan(2 * mu11 / (mu20 - mu02))
        orientation = np.degrees(theta)
    else:
        orientation = 0

    # Eccentricity
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

### B. Hu Moments (Shape Descriptor)

```python
def calculate_hu_moments(contour):
    """Calculate Hu moments for shape recognition"""

    M = cv2.moments(contour)
    hu_moments = cv2.HuMoments(M)

    # Log transform for better range
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

    return hu_moments.flatten()

# Compare shapes
hu1 = calculate_hu_moments(contour1)
hu2 = calculate_hu_moments(contour2)

distance = np.linalg.norm(hu1 - hu2)
print(f"Shape difference: {distance}")
```

---

## 11. Contour Convexity

### A. Check Convexity

```python
is_convex = cv2.isContourConvex(contour)

if is_convex:
    print("Contour is convex")
else:
    print("Contour has concavities")
```

### B. Convexity Defects

```python
# Get hull indices
hull = cv2.convexHull(contour, returnPoints=False)

# Calculate defects
defects = cv2.convexityDefects(contour, hull)

if defects is not None:
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # d is distance from farthest point to hull (in 1/256 pixels)
        depth = d / 256.0

        # Draw defects
        cv2.line(image, start, end, (0, 255, 0), 2)
        cv2.circle(image, far, 5, (0, 0, 255), -1)

        print(f"Defect depth: {depth:.2f}")
```

---

## 12. Complete Example

```python
class ContourAnalyzer:
    """Comprehensive contour analysis"""

    def __init__(self, min_area=500):
        self.min_area = min_area

    def analyze(self, image):
        """Analyze all contours in image"""

        # Prepare image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Analyze each contour
        results = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            # Calculate properties
            props = self._calculate_properties(contour)
            props['id'] = i
            props['contour'] = contour

            results.append(props)

        return results

    def _calculate_properties(self, contour):
        """Calculate all contour properties"""

        # Basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0

        # Extent
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Shape
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
        """Visualize analysis results"""

        vis = image.copy()

        for props in results:
            contour = props['contour']
            x, y, w, h = props['bbox']
            cx, cy = props['centroid']

            # Draw contour
            cv2.drawContours(vis, [contour], 0, (0, 255, 0), 2)

            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw centroid
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # Add label
            label = f"{props['shape']} ({props['area']:.0f})"
            cv2.putText(vis, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis


# Usage
analyzer = ContourAnalyzer(min_area=500)

# Analyze
results = analyzer.analyze(image)

# Print results
for props in results:
    print(f"Object {props['id']}:")
    print(f"  Shape: {props['shape']}")
    print(f"  Area: {props['area']:.0f}")
    print(f"  Aspect Ratio: {props['aspect_ratio']:.2f}")
    print(f"  Solidity: {props['solidity']:.2f}")

# Visualize
vis = analyzer.visualize(image, results)
cv2.imshow('Analysis', vis)
cv2.waitKey(0)
```

---

**Ngày tạo**: Tháng 1/2025
