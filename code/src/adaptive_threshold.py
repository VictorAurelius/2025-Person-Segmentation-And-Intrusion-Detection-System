"""
Adaptive thresholding module for handling variable lighting conditions.
"""

import cv2
import numpy as np
import logging
from typing import Tuple


class AdaptiveThreshold:
    """
    Adaptive thresholding class supporting multiple methods
    to handle non-uniform lighting conditions.
    """

    def __init__(self, method: str = "gaussian", block_size: int = 11, C: int = 2):
        """
        Initialize adaptive threshold processor.

        Args:
            method: Thresholding method ("gaussian", "mean", "otsu")
            block_size: Size of pixel neighborhood (must be odd)
            C: Constant subtracted from weighted mean
        """
        self.method = method.lower()
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.C = C

        logging.info(f"Initialized adaptive threshold: method={method}, "
                    f"block_size={self.block_size}, C={C}")

    def apply(self, image: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Apply adaptive thresholding to image.

        Args:
            image: Input image (grayscale or BGR)
            invert: Whether to invert the threshold

        Returns:
            Binary thresholded image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply method
        if self.method == "gaussian":
            result = self._adaptive_gaussian(gray, invert)
        elif self.method == "mean":
            result = self._adaptive_mean(gray, invert)
        elif self.method == "otsu":
            result = self._otsu(gray, invert)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return result

    def _adaptive_gaussian(self, gray: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Apply adaptive Gaussian thresholding.

        Args:
            gray: Grayscale image
            invert: Whether to invert threshold

        Returns:
            Binary image
        """
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            self.block_size,
            self.C
        )

        return result

    def _adaptive_mean(self, gray: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Apply adaptive mean thresholding.

        Args:
            gray: Grayscale image
            invert: Whether to invert threshold

        Returns:
            Binary image
        """
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            thresh_type,
            self.block_size,
            self.C
        )

        return result

    def _otsu(self, gray: np.ndarray, invert: bool = False) -> np.ndarray:
        """
        Apply Otsu's automatic thresholding.

        Args:
            gray: Grayscale image
            invert: Whether to invert threshold

        Returns:
            Binary image
        """
        # Apply Gaussian blur for better results
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

        _, result = cv2.threshold(
            blurred,
            0,
            255,
            thresh_type + cv2.THRESH_OTSU
        )

        return result

    def apply_with_preprocessing(self, image: np.ndarray,
                                invert: bool = False,
                                equalize: bool = True) -> np.ndarray:
        """
        Apply adaptive thresholding with preprocessing steps.

        Args:
            image: Input image
            invert: Whether to invert threshold
            equalize: Whether to apply histogram equalization

        Returns:
            Binary thresholded image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply histogram equalization if requested
        if equalize:
            gray = cv2.equalizeHist(gray)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply thresholding
        result = self.apply(gray, invert)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)

        return result


class CLAHEProcessor:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) processor
    for improving contrast in low-light conditions.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize CLAHE processor.

        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        logging.info(f"Initialized CLAHE: clip_limit={clip_limit}, "
                    f"tile_grid_size={tile_grid_size}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to image.

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # For color images, apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            result = self.clahe.apply(image)

        return result


class GammaCorrection:
    """
    Gamma correction for adjusting image brightness.
    """

    def __init__(self, gamma: float = 1.0):
        """
        Initialize gamma correction.

        Args:
            gamma: Gamma value (< 1 brightens, > 1 darkens)
        """
        self.gamma = gamma
        self._build_lookup_table()

        logging.info(f"Initialized gamma correction: gamma={gamma}")

    def _build_lookup_table(self) -> None:
        """Build lookup table for gamma correction."""
        self.lookup_table = np.array([
            ((i / 255.0) ** self.gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction to image.

        Args:
            image: Input image

        Returns:
            Corrected image
        """
        return cv2.LUT(image, self.lookup_table)

    def set_gamma(self, gamma: float) -> None:
        """
        Update gamma value.

        Args:
            gamma: New gamma value
        """
        self.gamma = gamma
        self._build_lookup_table()
        logging.info(f"Updated gamma to {gamma}")


class AutoThreshold:
    """
    Automatic threshold selection based on image statistics.
    """

    @staticmethod
    def calculate_optimal_threshold(image: np.ndarray) -> int:
        """
        Calculate optimal threshold using Otsu's method.

        Args:
            image: Input grayscale image

        Returns:
            Optimal threshold value
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Otsu's method
        threshold, _ = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return int(threshold)

    @staticmethod
    def calculate_adaptive_params(image: np.ndarray) -> Tuple[int, int]:
        """
        Calculate adaptive thresholding parameters based on image properties.

        Args:
            image: Input image

        Returns:
            Tuple of (block_size, C)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Adjust block size based on image size
        height, width = gray.shape
        min_dim = min(height, width)
        block_size = max(11, min(31, min_dim // 20))

        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1

        # Adjust C based on standard deviation
        C = max(2, min(10, int(std_intensity / 10)))

        logging.debug(f"Auto params: block_size={block_size}, C={C}, "
                     f"mean={mean_intensity:.1f}, std={std_intensity:.1f}")

        return block_size, C
