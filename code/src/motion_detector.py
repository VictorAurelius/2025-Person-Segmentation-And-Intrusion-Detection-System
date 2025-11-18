"""
Motion detection module for intrusion detection system.
Implements frame differencing and background subtraction methods.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional


class MotionDetector:
    """
    Motion detector class supporting multiple detection methods:
    - Frame differencing
    - MOG2 background subtraction
    - KNN background subtraction
    """

    def __init__(self, method: str = "MOG2", history: int = 500,
                 threshold: int = 16, detect_shadows: bool = True):
        """
        Initialize motion detector.

        Args:
            method: Detection method ("MOG2", "KNN", "frame_diff")
            history: Number of frames for background learning
            threshold: Threshold for frame differencing
            detect_shadows: Whether to detect shadows
        """
        self.method = method.upper()
        self.history = history
        self.threshold = threshold
        self.detect_shadows = detect_shadows
        self.prev_frame = None

        # Initialize background subtractor based on method
        if self.method == "MOG2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=threshold,
                detectShadows=detect_shadows
            )
            logging.info("Initialized MOG2 background subtractor")

        elif self.method == "KNN":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=threshold * 10,
                detectShadows=detect_shadows
            )
            logging.info("Initialized KNN background subtractor")

        elif self.method == "FRAME_DIFF":
            self.bg_subtractor = None
            logging.info("Initialized frame differencing method")

        else:
            raise ValueError(f"Unknown method: {method}. Use 'MOG2', 'KNN', or 'frame_diff'")

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect motion in the given frame.

        Args:
            frame: Input frame (BGR)

        Returns:
            Tuple of (foreground mask, processed mask)
        """
        if self.method == "FRAME_DIFF":
            return self._frame_differencing(frame)
        else:
            return self._background_subtraction(frame)

    def _background_subtraction(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply background subtraction to detect moving objects.

        Args:
            frame: Input frame

        Returns:
            Tuple of (foreground mask, processed mask)
        """
        # Apply background subtractor
        fg_mask = self.bg_subtractor.apply(frame)

        # If shadows are detected, they are marked as 127 (gray)
        # We want to treat them as background (0)
        if self.detect_shadows:
            fg_mask[fg_mask == 127] = 0

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Opening: removes noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Closing: fills holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Dilate to make detected regions more prominent
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        return fg_mask, fg_mask

    def _frame_differencing(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply frame differencing to detect motion.

        Args:
            frame: Input frame

        Returns:
            Tuple of (difference image, thresholded mask)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If this is the first frame, initialize and return empty mask
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray), np.zeros_like(gray)

        # Compute absolute difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)

        # Threshold the difference
        _, thresh = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Opening: removes noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Closing: fills holes
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Dilate
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Update previous frame
        self.prev_frame = gray

        return frame_diff, thresh

    def get_contours(self, mask: np.ndarray,
                    min_area: int = 500) -> list:
        """
        Find contours in the mask.

        Args:
            mask: Binary mask
            min_area: Minimum contour area to keep

        Returns:
            List of contours
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        logging.debug(f"Found {len(contours)} contours, {len(filtered_contours)} after filtering")

        return filtered_contours

    def reset_background_model(self) -> None:
        """
        Reset the background model.
        Useful when there's a significant scene change.
        """
        if self.method == "MOG2":
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.threshold,
                detectShadows=self.detect_shadows
            )
        elif self.method == "KNN":
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=self.threshold * 10,
                detectShadows=self.detect_shadows
            )
        elif self.method == "FRAME_DIFF":
            self.prev_frame = None

        logging.info("Background model reset")

    def get_background_model(self) -> Optional[np.ndarray]:
        """
        Get the current background model.

        Returns:
            Background image or None if not available
        """
        if self.method in ["MOG2", "KNN"]:
            return self.bg_subtractor.getBackgroundImage()
        return None


class ThreeFrameDifferencing:
    """
    Three-frame differencing for more robust motion detection.
    """

    def __init__(self, threshold: int = 25):
        """
        Initialize three-frame differencing.

        Args:
            threshold: Threshold for binary conversion
        """
        self.threshold = threshold
        self.frame_buffer = []

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect motion using three consecutive frames.

        Args:
            frame: Input frame

        Returns:
            Motion mask
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Add to buffer
        self.frame_buffer.append(gray)

        # Keep only last 3 frames
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        # Need at least 3 frames
        if len(self.frame_buffer) < 3:
            return np.zeros_like(gray)

        # Compute differences
        diff1 = cv2.absdiff(self.frame_buffer[0], self.frame_buffer[1])
        diff2 = cv2.absdiff(self.frame_buffer[1], self.frame_buffer[2])

        # Threshold differences
        _, thresh1 = cv2.threshold(diff1, self.threshold, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(diff2, self.threshold, 255, cv2.THRESH_BINARY)

        # AND operation: motion detected in both differences
        motion = cv2.bitwise_and(thresh1, thresh2)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)

        return motion
