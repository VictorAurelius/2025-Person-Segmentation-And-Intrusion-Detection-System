# Code Documentation

## Module Overview

This directory contains the implementation of the intrusion detection system.

---

## Source Modules (`src/`)

### `main.py`
Main application entry point. Orchestrates the entire detection pipeline.

**Key Classes:**
- `IntrusionDetectionSystem`: Main system class

**Functions:**
- `process_video()`: Process video through detection pipeline
- `parse_arguments()`: Handle command-line arguments

**Usage:**
```bash
python src/main.py --source video.mp4
```

---

### `motion_detector.py`
Motion detection using background subtraction and frame differencing.

**Key Classes:**
- `MotionDetector`: Main motion detection class
  - Methods: MOG2, KNN, Frame Differencing
- `ThreeFrameDifferencing`: Three-frame differencing for robustness

**Key Methods:**
- `detect()`: Detect motion in frame
- `get_contours()`: Extract motion contours
- `reset_background_model()`: Reset background

**Example:**
```python
from motion_detector import MotionDetector

detector = MotionDetector(method="MOG2", threshold=16)
fg_mask, processed = detector.detect(frame)
contours = detector.get_contours(processed, min_area=1000)
```

---

### `adaptive_threshold.py`
Adaptive thresholding for variable lighting conditions.

**Key Classes:**
- `AdaptiveThreshold`: Adaptive thresholding (Gaussian, Mean, Otsu)
- `CLAHEProcessor`: Contrast Limited Adaptive Histogram Equalization
- `GammaCorrection`: Brightness adjustment
- `AutoThreshold`: Automatic threshold calculation

**Key Methods:**
- `apply()`: Apply adaptive thresholding
- `apply_with_preprocessing()`: Threshold with pre-processing

**Example:**
```python
from adaptive_threshold import AdaptiveThreshold

threshold = AdaptiveThreshold(method="gaussian", block_size=11, C=2)
binary = threshold.apply(frame, invert=True)
```

---

### `edge_detector.py`
Edge detection using various algorithms.

**Key Classes:**
- `EdgeDetector`: Edge detection (Canny, Sobel, Prewitt, Scharr)
- `EdgeLinking`: Edge fragment linking

**Key Methods:**
- `detect()`: Detect edges
- `detect_with_direction()`: Get edges with gradient direction
- `hysteresis_threshold()`: Apply hysteresis thresholding

**Example:**
```python
from edge_detector import EdgeDetector

detector = EdgeDetector(method="canny", low_threshold=50, high_threshold=150)
edges = detector.detect(frame)
```

---

### `region_grower.py`
Region growing segmentation.

**Key Classes:**
- `RegionGrower`: Basic region growing
- `SeededRegionGrowing`: Multi-region seeded growing
- `WatershedSegmentation`: Watershed segmentation

**Key Methods:**
- `grow()`: Grow regions from seeds
- `grow_with_gradient()`: Region growing with gradient

**Example:**
```python
from region_grower import RegionGrower

grower = RegionGrower(threshold=10.0, connectivity=8)
seeds = [(100, 100), (200, 200)]
mask = grower.grow(gray_image, seeds)
```

---

### `intrusion_detector.py`
Intrusion detection logic for ROI monitoring.

**Key Classes:**
- `IntrusionDetector`: Main intrusion detection
- `IntrusionZone`: Helper for zone management

**Key Methods:**
- `detect_intrusions()`: Check for intrusions
- `check_point_in_roi()`: Point-in-polygon test
- `reset_tracking()`: Clear tracking data

**Example:**
```python
from intrusion_detector import IntrusionDetector

detector = IntrusionDetector(
    roi_definitions=rois,
    overlap_threshold=0.3,
    time_threshold=1.0
)
flags, details = detector.detect_intrusions(contours, time.time())
```

---

### `alert_system.py`
Alert generation and logging.

**Key Classes:**
- `AlertSystem`: Visual, audio, and logging alerts
- `AlertNotifier`: Extended notifications (email, webhook)

**Key Methods:**
- `trigger_alert()`: Trigger alert for intrusions
- `add_info_overlay()`: Add info overlay to frame
- `get_alert_summary()`: Get alert statistics

**Example:**
```python
from alert_system import AlertSystem

alerts = AlertSystem(
    visual=True,
    audio=True,
    log_file="alerts.log"
)
result = alerts.trigger_alert(frame, intrusion_details, frame_num)
```

---

### `utils.py`
Utility functions for common operations.

**Key Functions:**
- `load_config()`: Load YAML configuration
- `load_roi_definitions()`: Load ROI JSON
- `draw_roi()`: Draw ROI overlays
- `draw_bounding_boxes()`: Draw detection boxes
- `calculate_iou()`: Calculate IoU between boxes
- `calculate_overlap_percentage()`: Calculate overlap ratio

**Example:**
```python
from utils import load_config, load_roi_definitions

config = load_config("config/config.yaml")
rois = load_roi_definitions("data/roi/restricted_area.json")
```

---

## Tools (`tools/`)

### `roi_selector.py`
Interactive ROI definition tool.

**Usage:**
```bash
python tools/roi_selector.py --video data/input/video.mp4
```

**Controls:**
- Left click: Add point
- Right click: Finish ROI
- 'c': Clear points
- 'd': Delete last ROI
- 's': Save and exit
- 'q': Quit without saving

---

## Configuration (`config/`)

### `config.yaml`
Main configuration file. See main README for full details.

**Sections:**
- `video`: Video source settings
- `motion`: Motion detection parameters
- `threshold`: Adaptive thresholding settings
- `edge`: Edge detection settings
- `morphology`: Morphological operation settings
- `intrusion`: Intrusion detection parameters
- `alert`: Alert system configuration
- `output`: Output settings

---

## Data Organization (`data/`)

```
data/
├── input/           # Place input videos here
├── output/          # Generated outputs
│   ├── result.mp4           # Processed video
│   ├── alerts.log           # Alert log
│   └── screenshots/         # Alert screenshots
└── roi/             # ROI definitions
    └── restricted_area.json
```

---

## Dependencies (`requirements.txt`)

- `opencv-python>=4.8.0`: Computer vision
- `numpy>=1.24.0`: Array operations
- `scikit-image>=0.21.0`: Image processing
- `matplotlib>=3.7.0`: Visualization
- `pyyaml>=6.0`: YAML parsing
- `pytest>=7.4.0`: Testing
- `scipy>=1.10.0`: Scientific computing

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Module Dependencies

```
main.py
  ├── utils.py
  ├── motion_detector.py
  ├── adaptive_threshold.py
  ├── edge_detector.py
  ├── region_grower.py
  ├── intrusion_detector.py
  └── alert_system.py
```

All modules are loosely coupled and can be used independently.

---

## Code Style

- **PEP 8** compliant
- **Type hints** for function signatures
- **Docstrings** for all classes and methods
- **Logging** for debugging and info
- **Error handling** with try-except blocks

---

## Testing

Run tests (when available):
```bash
pytest tests/
```

---

## Performance Optimization Tips

1. **Reduce resolution**: Resize frames to 640x360 for faster processing
2. **Skip frames**: Process every 2nd or 3rd frame
3. **Disable display**: Use `--no-display` for headless processing
4. **Use frame_diff**: Faster than MOG2/KNN
5. **Simplify ROI**: Fewer polygon points = faster overlap calculation

---

## Adding Custom Modules

To add a new module:

1. Create file in `src/` directory
2. Follow existing structure (classes, docstrings, type hints)
3. Add import to `main.py` if needed
4. Update this README

---

## Debugging

Enable debug logging:
```bash
python src/main.py --debug
```

Check logs:
```bash
tail -f data/output/alerts.log
```

Test individual modules:
```python
# Test motion detection only
from motion_detector import MotionDetector
detector = MotionDetector()
# ... test code
```

---

## Common Issues

**Import errors**: Ensure virtual environment is activated
**Video not opening**: Check file path and format
**Slow performance**: Reduce resolution or use simpler algorithms
**No alerts**: Check ROI definitions and thresholds

---

## Version

**Version**: 1.0
**Date**: January 2025
**Python**: 3.8+
