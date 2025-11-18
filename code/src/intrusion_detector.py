"""
Intrusion detection module for detecting objects entering restricted areas.
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Any
from collections import defaultdict


class IntrusionDetector:
    """
    Intrusion detector class for monitoring restricted areas.
    """

    def __init__(self, roi_definitions: List[Dict[str, Any]],
                 overlap_threshold: float = 0.3,
                 time_threshold: float = 1.0,
                 min_object_area: int = 1000):
        """
        Initialize intrusion detector.

        Args:
            roi_definitions: List of ROI definitions
            overlap_threshold: Minimum overlap ratio to trigger (0-1)
            time_threshold: Minimum time (seconds) object must be in ROI
            min_object_area: Minimum object area to consider (pixels)
        """
        self.roi_definitions = roi_definitions
        self.overlap_threshold = overlap_threshold
        self.time_threshold = time_threshold
        self.min_object_area = min_object_area

        # Track objects in restricted areas
        self.intrusion_tracking = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'roi_name': None
        })

        logging.info(f"Initialized intrusion detector: "
                    f"overlap={overlap_threshold}, time={time_threshold}s, "
                    f"min_area={min_object_area}")

    def detect_intrusions(self, contours: List[np.ndarray],
                         frame_time: float) -> Tuple[List[bool], List[Dict[str, Any]]]:
        """
        Detect intrusions for given contours.

        Args:
            contours: List of detected object contours
            frame_time: Current frame timestamp

        Returns:
            Tuple of (intrusion flags, intrusion details)
        """
        intrusion_flags = []
        intrusion_details = []

        for i, contour in enumerate(contours):
            # Check minimum area
            area = cv2.contourArea(contour)
            if area < self.min_object_area:
                intrusion_flags.append(False)
                continue

            # Check each ROI
            is_intrusion = False
            intrusion_info = None

            for roi in self.roi_definitions:
                overlap = self._calculate_overlap(contour, roi)

                if overlap >= self.overlap_threshold:
                    # Track this intrusion
                    intrusion_key = self._get_intrusion_key(contour, roi)

                    if intrusion_key not in self.intrusion_tracking:
                        # First time seeing this intrusion
                        self.intrusion_tracking[intrusion_key] = {
                            'first_seen': frame_time,
                            'last_seen': frame_time,
                            'roi_name': roi['name']
                        }
                    else:
                        # Update last seen time
                        self.intrusion_tracking[intrusion_key]['last_seen'] = frame_time

                    # Check if intrusion has persisted long enough
                    tracking_data = self.intrusion_tracking[intrusion_key]
                    duration = frame_time - tracking_data['first_seen']

                    if duration >= self.time_threshold:
                        is_intrusion = True
                        x, y, w, h = cv2.boundingRect(contour)
                        intrusion_info = {
                            'roi_name': roi['name'],
                            'overlap': overlap,
                            'duration': duration,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'center': (x + w // 2, y + h // 2)
                        }
                        break

            intrusion_flags.append(is_intrusion)
            if intrusion_info:
                intrusion_details.append(intrusion_info)

        # Clean up old tracking data
        self._cleanup_old_tracking(frame_time)

        return intrusion_flags, intrusion_details

    def _calculate_overlap(self, contour: np.ndarray,
                          roi_definition: Dict[str, Any]) -> float:
        """
        Calculate overlap percentage between contour and ROI.

        Args:
            contour: Object contour
            roi_definition: ROI definition

        Returns:
            Overlap ratio (0-1)
        """
        # Get bounding rect for faster computation
        x, y, w, h = cv2.boundingRect(contour)

        # Create masks
        height = y + h + 10
        width = x + w + 10

        # Contour mask
        mask_contour = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask_contour, [contour], -1, 255, -1)

        # ROI mask
        mask_roi = np.zeros((height, width), dtype=np.uint8)

        if roi_definition['type'] == 'polygon':
            points = np.array(roi_definition['points'], dtype=np.int32)
            cv2.fillPoly(mask_roi, [points], 255)
        elif roi_definition['type'] == 'rectangle':
            x_roi, y_roi = roi_definition['x'], roi_definition['y']
            w_roi, h_roi = roi_definition['width'], roi_definition['height']
            cv2.rectangle(mask_roi, (x_roi, y_roi),
                        (x_roi + w_roi, y_roi + h_roi), 255, -1)

        # Calculate intersection
        intersection = cv2.bitwise_and(mask_contour, mask_roi)
        intersection_area = cv2.countNonZero(intersection)
        contour_area = cv2.countNonZero(mask_contour)

        if contour_area == 0:
            return 0.0

        overlap = intersection_area / contour_area

        return overlap

    def _get_intrusion_key(self, contour: np.ndarray,
                          roi_definition: Dict[str, Any]) -> str:
        """
        Generate unique key for tracking intrusion.

        Args:
            contour: Object contour
            roi_definition: ROI definition

        Returns:
            Unique tracking key
        """
        # Use centroid and ROI name as key
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

        # Round to grid for more stable tracking
        grid_size = 50
        cx_grid = (cx // grid_size) * grid_size
        cy_grid = (cy // grid_size) * grid_size

        return f"{roi_definition['name']}_{cx_grid}_{cy_grid}"

    def _cleanup_old_tracking(self, current_time: float, timeout: float = 5.0) -> None:
        """
        Remove old tracking entries.

        Args:
            current_time: Current timestamp
            timeout: Timeout for old entries (seconds)
        """
        keys_to_remove = []

        for key, data in self.intrusion_tracking.items():
            if current_time - data['last_seen'] > timeout:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.intrusion_tracking[key]
            logging.debug(f"Removed old tracking entry: {key}")

    def check_point_in_roi(self, point: Tuple[int, int],
                          roi_name: str = None) -> bool:
        """
        Check if a point is inside ROI(s).

        Args:
            point: Point coordinates (x, y)
            roi_name: Specific ROI name to check (None = check all)

        Returns:
            True if point is in ROI
        """
        rois_to_check = self.roi_definitions

        if roi_name:
            rois_to_check = [roi for roi in self.roi_definitions
                           if roi['name'] == roi_name]

        for roi in rois_to_check:
            if roi['type'] == 'polygon':
                points_array = np.array(roi['points'], dtype=np.int32)
                if cv2.pointPolygonTest(points_array, point, False) >= 0:
                    return True

            elif roi['type'] == 'rectangle':
                x_roi, y_roi = roi['x'], roi['y']
                w_roi, h_roi = roi['width'], roi['height']
                x, y = point

                if (x_roi <= x <= x_roi + w_roi and
                    y_roi <= y <= y_roi + h_roi):
                    return True

        return False

    def get_roi_by_name(self, name: str) -> Dict[str, Any]:
        """
        Get ROI definition by name.

        Args:
            name: ROI name

        Returns:
            ROI definition dictionary or None
        """
        for roi in self.roi_definitions:
            if roi['name'] == name:
                return roi
        return None

    def reset_tracking(self) -> None:
        """
        Reset all intrusion tracking data.
        """
        self.intrusion_tracking.clear()
        logging.info("Intrusion tracking data reset")

    def get_active_intrusions(self) -> List[Dict[str, Any]]:
        """
        Get list of currently active intrusions.

        Returns:
            List of active intrusion data
        """
        return [
            {
                'key': key,
                'roi_name': data['roi_name'],
                'duration': time.time() - data['first_seen']
            }
            for key, data in self.intrusion_tracking.items()
        ]


class IntrusionZone:
    """
    Helper class for defining and managing intrusion zones.
    """

    def __init__(self, name: str, zone_type: str = "polygon"):
        """
        Initialize intrusion zone.

        Args:
            name: Zone name
            zone_type: Type of zone ("polygon" or "rectangle")
        """
        self.name = name
        self.zone_type = zone_type
        self.points = []
        self.color = [255, 0, 0]  # Default red

    def add_point(self, x: int, y: int) -> None:
        """
        Add a point to the zone.

        Args:
            x, y: Point coordinates
        """
        self.points.append([x, y])

    def set_rectangle(self, x: int, y: int, width: int, height: int) -> None:
        """
        Set zone as rectangle.

        Args:
            x, y: Top-left corner
            width, height: Rectangle dimensions
        """
        self.zone_type = "rectangle"
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def set_color(self, color: List[int]) -> None:
        """
        Set zone color.

        Args:
            color: BGR color [B, G, R]
        """
        self.color = color

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert zone to dictionary format.

        Returns:
            Zone definition dictionary
        """
        if self.zone_type == "polygon":
            return {
                "name": self.name,
                "type": "polygon",
                "points": self.points,
                "color": self.color
            }
        else:
            return {
                "name": self.name,
                "type": "rectangle",
                "x": self.x,
                "y": self.y,
                "width": self.width,
                "height": self.height,
                "color": self.color
            }

    def contains_point(self, x: int, y: int) -> bool:
        """
        Check if point is inside zone.

        Args:
            x, y: Point coordinates

        Returns:
            True if inside zone
        """
        if self.zone_type == "polygon":
            points_array = np.array(self.points, dtype=np.int32)
            return cv2.pointPolygonTest(points_array, (x, y), False) >= 0
        else:
            return (self.x <= x <= self.x + self.width and
                   self.y <= y <= self.y + self.height)

    def draw(self, frame: np.ndarray, thickness: int = 2) -> np.ndarray:
        """
        Draw zone on frame.

        Args:
            frame: Input frame
            thickness: Line thickness

        Returns:
            Frame with zone drawn
        """
        result = frame.copy()

        if self.zone_type == "polygon" and len(self.points) >= 3:
            points_array = np.array(self.points, dtype=np.int32)
            cv2.polylines(result, [points_array], True,
                        tuple(self.color), thickness)
        elif self.zone_type == "rectangle":
            cv2.rectangle(result, (self.x, self.y),
                        (self.x + self.width, self.y + self.height),
                        tuple(self.color), thickness)

        return result
