"""
Utility functions for the intrusion detection system.
"""

import cv2
import numpy as np
import yaml
import json
import logging
from typing import Dict, List, Tuple, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the config YAML file

    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise


def load_roi_definitions(roi_path: str) -> List[Dict[str, Any]]:
    """
    Load ROI (Region of Interest) definitions from JSON file.

    Args:
        roi_path: Path to the ROI JSON file

    Returns:
        List of ROI definitions
    """
    try:
        with open(roi_path, 'r') as f:
            roi_data = json.load(f)
        logging.info(f"Loaded {len(roi_data['restricted_areas'])} ROI definitions")
        return roi_data['restricted_areas']
    except Exception as e:
        logging.error(f"Error loading ROI definitions: {e}")
        raise


def draw_roi(frame: np.ndarray, roi_definitions: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw ROI areas on the frame.

    Args:
        frame: Input frame
        roi_definitions: List of ROI definitions

    Returns:
        Frame with ROI overlay
    """
    overlay = frame.copy()

    for roi in roi_definitions:
        color = tuple(roi.get('color', [255, 0, 0]))

        if roi['type'] == 'polygon':
            points = np.array(roi['points'], dtype=np.int32)
            cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(overlay, [points], color=(*color[:3], 50))

        elif roi['type'] == 'rectangle':
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

    # Blend overlay with original frame
    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Draw labels
    for roi in roi_definitions:
        if roi['type'] == 'polygon':
            points = np.array(roi['points'], dtype=np.int32)
            cx, cy = points.mean(axis=0).astype(int)
        else:
            cx = roi['x'] + roi['width'] // 2
            cy = roi['y'] + roi['height'] // 2

        cv2.putText(result, roi['name'], (cx - 30, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result


def calculate_iou(box1: Tuple[int, int, int, int],
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box (x, y, w, h)
        box2: Second bounding box (x, y, w, h)

    Returns:
        IoU value (0.0 to 1.0)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        point: Point coordinates (x, y)
        polygon: Array of polygon vertices

    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def calculate_overlap_percentage(contour: np.ndarray,
                                 roi_definition: Dict[str, Any]) -> float:
    """
    Calculate the percentage of contour that overlaps with ROI.

    Args:
        contour: Detected object contour
        roi_definition: ROI definition dictionary

    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    # Get contour bounding box
    x, y, w, h = cv2.boundingRect(contour)
    contour_area = cv2.contourArea(contour)

    if contour_area == 0:
        return 0.0

    # Create mask for contour
    mask_contour = np.zeros((y + h + 10, x + w + 10), dtype=np.uint8)
    cv2.drawContours(mask_contour, [contour], -1, 255, -1)

    # Create mask for ROI
    mask_roi = np.zeros_like(mask_contour)

    if roi_definition['type'] == 'polygon':
        points = np.array(roi_definition['points'], dtype=np.int32)
        cv2.fillPoly(mask_roi, [points], 255)
    elif roi_definition['type'] == 'rectangle':
        x_roi, y_roi = roi_definition['x'], roi_definition['y']
        w_roi, h_roi = roi_definition['width'], roi_definition['height']
        cv2.rectangle(mask_roi, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), 255, -1)

    # Calculate intersection
    intersection = cv2.bitwise_and(mask_contour, mask_roi)
    intersection_area = cv2.countNonZero(intersection)

    # Calculate overlap percentage
    overlap = intersection_area / contour_area if contour_area > 0 else 0.0

    return overlap


def draw_bounding_boxes(frame: np.ndarray,
                       contours: List[np.ndarray],
                       is_intrusion: List[bool] = None,
                       min_area: int = 1000) -> np.ndarray:
    """
    Draw bounding boxes around detected objects.

    Args:
        frame: Input frame
        contours: List of contours to draw
        is_intrusion: List of boolean flags indicating intrusion
        min_area: Minimum contour area to draw

    Returns:
        Frame with bounding boxes
    """
    result = frame.copy()

    if is_intrusion is None:
        is_intrusion = [False] * len(contours)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Choose color based on intrusion status
        color = (0, 0, 255) if is_intrusion[i] else (0, 255, 0)
        label = "INTRUSION" if is_intrusion[i] else "DETECTED"

        # Draw bounding box
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x, y - label_size[1] - 10),
                     (x + label_size[0], y), color, -1)
        cv2.putText(result, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return result


def add_text_overlay(frame: np.ndarray,
                     text: str,
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 0.7,
                     color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Add text overlay to frame.

    Args:
        frame: Input frame
        text: Text to display
        position: Text position (x, y)
        font_scale: Font scale
        color: Text color (B, G, R)

    Returns:
        Frame with text overlay
    """
    result = frame.copy()
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, (0, 0, 0), 4)  # Shadow
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, 2)  # Text
    return result


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
