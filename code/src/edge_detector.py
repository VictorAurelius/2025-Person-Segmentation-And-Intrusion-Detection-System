"""
Edge detection module for intrusion detection system.
Implements various edge detection algorithms.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional


class EdgeDetector:
    """
    Edge detector class supporting multiple algorithms:
    - Canny edge detection
    - Sobel operator
    - Prewitt operator
    - Scharr operator
    """

    def __init__(self, method: str = "canny",
                 low_threshold: int = 50,
                 high_threshold: int = 150):
        """
        Initialize edge detector.

        Args:
            method: Edge detection method ("canny", "sobel", "prewitt", "scharr")
            low_threshold: Lower threshold for Canny or Sobel magnitude
            high_threshold: Upper threshold for Canny
        """
        self.method = method.lower()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        logging.info(f"Initialized edge detector: method={method}, "
                    f"low={low_threshold}, high={high_threshold}")

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image.

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            Edge map (binary image)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply edge detection method
        if self.method == "canny":
            edges = self._canny(gray)
        elif self.method == "sobel":
            edges = self._sobel(gray)
        elif self.method == "prewitt":
            edges = self._prewitt(gray)
        elif self.method == "scharr":
            edges = self._scharr(gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return edges

    def _canny(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection.

        Args:
            gray: Grayscale image

        Returns:
            Edge map
        """
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold,
                         apertureSize=3, L2gradient=True)
        return edges

    def _sobel(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Sobel edge detection.

        Args:
            gray: Grayscale image

        Returns:
            Edge map
        """
        # Compute gradients in x and y directions
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))

        # Apply threshold
        _, edges = cv2.threshold(magnitude, self.low_threshold, 255,
                                cv2.THRESH_BINARY)

        return edges

    def _prewitt(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Prewitt edge detection.

        Args:
            gray: Grayscale image

        Returns:
            Edge map
        """
        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float32)

        kernel_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]], dtype=np.float32)

        # Apply filters
        grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))

        # Apply threshold
        _, edges = cv2.threshold(magnitude, self.low_threshold, 255,
                                cv2.THRESH_BINARY)

        return edges

    def _scharr(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Scharr edge detection (more accurate than Sobel).

        Args:
            gray: Grayscale image

        Returns:
            Edge map
        """
        # Compute gradients using Scharr operator
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))

        # Apply threshold
        _, edges = cv2.threshold(magnitude, self.low_threshold, 255,
                                cv2.THRESH_BINARY)

        return edges

    def detect_with_direction(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect edges with gradient direction.

        Args:
            image: Input image

        Returns:
            Tuple of (edge map, gradient direction)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Compute gradients
        if self.method in ["sobel", "canny"]:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        elif self.method == "scharr":
            grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        else:
            # Default to Sobel
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Normalize magnitude
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))

        # Threshold to get edge map
        _, edges = cv2.threshold(magnitude, self.low_threshold, 255,
                                cv2.THRESH_BINARY)

        return edges, direction

    def apply_non_maximum_suppression(self, magnitude: np.ndarray,
                                     direction: np.ndarray) -> np.ndarray:
        """
        Apply non-maximum suppression to thin edges.

        Args:
            magnitude: Gradient magnitude
            direction: Gradient direction

        Returns:
            Suppressed edge map
        """
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # Convert direction to degrees
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                q = 255
                r = 255

                # Determine neighbors based on angle
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                # Suppress if not local maximum
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        return suppressed

    def hysteresis_threshold(self, image: np.ndarray,
                            low: Optional[int] = None,
                            high: Optional[int] = None) -> np.ndarray:
        """
        Apply hysteresis thresholding (used in Canny).

        Args:
            image: Input edge magnitude image
            low: Low threshold (defaults to self.low_threshold)
            high: High threshold (defaults to self.high_threshold)

        Returns:
            Binary edge map
        """
        if low is None:
            low = self.low_threshold
        if high is None:
            high = self.high_threshold

        # Create strong, weak, and non-edge regions
        strong = 255
        weak = 75

        result = np.zeros_like(image)
        strong_i, strong_j = np.where(image >= high)
        weak_i, weak_j = np.where((image >= low) & (image < high))

        result[strong_i, strong_j] = strong
        result[weak_i, weak_j] = weak

        # Connect weak edges to strong edges
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                if result[i, j] == weak:
                    # Check if connected to strong edge
                    if ((result[i + 1, j - 1] == strong) or
                        (result[i + 1, j] == strong) or
                        (result[i + 1, j + 1] == strong) or
                        (result[i, j - 1] == strong) or
                        (result[i, j + 1] == strong) or
                        (result[i - 1, j - 1] == strong) or
                        (result[i - 1, j] == strong) or
                        (result[i - 1, j + 1] == strong)):
                        result[i, j] = strong
                    else:
                        result[i, j] = 0

        # Convert to binary
        result[result == weak] = 0
        result[result == strong] = 255

        return result.astype(np.uint8)


class EdgeLinking:
    """
    Edge linking utility for connecting edge fragments.
    """

    @staticmethod
    def link_edges(edges: np.ndarray, max_gap: int = 5) -> np.ndarray:
        """
        Link edge fragments by filling small gaps.

        Args:
            edges: Binary edge map
            max_gap: Maximum gap to fill (in pixels)

        Returns:
            Linked edge map
        """
        # Use morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (max_gap, max_gap))
        linked = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Thin the result to get single-pixel edges
        linked = cv2.ximgproc.thinning(linked)

        return linked

    @staticmethod
    def remove_small_edges(edges: np.ndarray, min_length: int = 10) -> np.ndarray:
        """
        Remove small edge fragments.

        Args:
            edges: Binary edge map
            min_length: Minimum edge length to keep

        Returns:
            Filtered edge map
        """
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

        # Filter by arc length
        result = np.zeros_like(edges)
        for contour in contours:
            if cv2.arcLength(contour, False) >= min_length:
                cv2.drawContours(result, [contour], -1, 255, 1)

        return result
