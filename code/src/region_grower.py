"""
Region growing module for image segmentation.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from collections import deque


class RegionGrower:
    """
    Region growing segmentation class.
    Grows regions from seed points based on similarity criteria.
    """

    def __init__(self, threshold: float = 10.0, connectivity: int = 8):
        """
        Initialize region grower.

        Args:
            threshold: Similarity threshold for region growing
            connectivity: Connectivity type (4 or 8)
        """
        self.threshold = threshold
        self.connectivity = connectivity

        logging.info(f"Initialized region grower: threshold={threshold}, "
                    f"connectivity={connectivity}")

    def grow(self, image: np.ndarray,
            seeds: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        Perform region growing segmentation.

        Args:
            image: Input grayscale image
            seeds: List of seed points (x, y). If None, auto-select seeds.

        Returns:
            Segmentation mask
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Auto-select seeds if not provided
        if seeds is None:
            seeds = self._auto_select_seeds(gray)

        # Initialize segmentation mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        visited = np.zeros(gray.shape, dtype=bool)

        # Grow region from each seed
        region_id = 1
        for seed in seeds:
            if not visited[seed[1], seed[0]]:
                self._grow_from_seed(gray, seed, mask, visited, region_id)
                region_id += 1

        return mask

    def _grow_from_seed(self, image: np.ndarray,
                       seed: Tuple[int, int],
                       mask: np.ndarray,
                       visited: np.ndarray,
                       region_id: int) -> None:
        """
        Grow a single region from a seed point.

        Args:
            image: Input image
            seed: Seed point (x, y)
            mask: Segmentation mask to update
            visited: Visited pixels tracker
            region_id: ID for this region
        """
        height, width = image.shape
        seed_x, seed_y = seed

        # Check bounds
        if (seed_x < 0 or seed_x >= width or
            seed_y < 0 or seed_y >= height):
            return

        # Get seed intensity
        seed_intensity = float(image[seed_y, seed_x])

        # BFS queue
        queue = deque([seed])
        visited[seed_y, seed_x] = True
        mask[seed_y, seed_x] = region_id

        while queue:
            x, y = queue.popleft()
            current_intensity = float(image[y, x])

            # Get neighbors based on connectivity
            neighbors = self._get_neighbors(x, y, width, height)

            for nx, ny in neighbors:
                if not visited[ny, nx]:
                    neighbor_intensity = float(image[ny, nx])

                    # Check similarity criterion
                    if abs(neighbor_intensity - seed_intensity) <= self.threshold:
                        visited[ny, nx] = True
                        mask[ny, nx] = region_id
                        queue.append((nx, ny))

    def _get_neighbors(self, x: int, y: int,
                      width: int, height: int) -> List[Tuple[int, int]]:
        """
        Get neighboring pixels based on connectivity.

        Args:
            x, y: Current pixel position
            width, height: Image dimensions

        Returns:
            List of neighbor coordinates
        """
        neighbors = []

        # 4-connectivity
        if self.connectivity == 4:
            offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        # 8-connectivity
        else:
            offsets = [(-1, -1), (0, -1), (1, -1),
                      (-1, 0), (1, 0),
                      (-1, 1), (0, 1), (1, 1)]

        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))

        return neighbors

    def _auto_select_seeds(self, image: np.ndarray,
                          grid_size: int = 20) -> List[Tuple[int, int]]:
        """
        Automatically select seed points on a grid.

        Args:
            image: Input image
            grid_size: Spacing between seeds

        Returns:
            List of seed points
        """
        height, width = image.shape
        seeds = []

        for y in range(grid_size // 2, height, grid_size):
            for x in range(grid_size // 2, width, grid_size):
                seeds.append((x, y))

        logging.debug(f"Auto-selected {len(seeds)} seed points")
        return seeds

    def grow_with_gradient(self, image: np.ndarray,
                          gradient: np.ndarray,
                          seeds: Optional[List[Tuple[int, int]]] = None,
                          gradient_weight: float = 0.5) -> np.ndarray:
        """
        Perform region growing with gradient information.

        Args:
            image: Input image
            gradient: Gradient magnitude image
            seeds: Seed points
            gradient_weight: Weight for gradient similarity (0-1)

        Returns:
            Segmentation mask
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Normalize gradient to 0-255
        gradient_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
        gradient_norm = gradient_norm.astype(np.float32)

        # Auto-select seeds if not provided
        if seeds is None:
            seeds = self._auto_select_seeds(gray)

        # Initialize masks
        mask = np.zeros(gray.shape, dtype=np.uint8)
        visited = np.zeros(gray.shape, dtype=bool)

        # Grow regions
        region_id = 1
        for seed in seeds:
            if not visited[seed[1], seed[0]]:
                self._grow_with_gradient_from_seed(
                    gray, gradient_norm, seed, mask, visited,
                    region_id, gradient_weight
                )
                region_id += 1

        return mask

    def _grow_with_gradient_from_seed(self, image: np.ndarray,
                                     gradient: np.ndarray,
                                     seed: Tuple[int, int],
                                     mask: np.ndarray,
                                     visited: np.ndarray,
                                     region_id: int,
                                     gradient_weight: float) -> None:
        """
        Grow region with gradient consideration.

        Args:
            image: Input image
            gradient: Gradient magnitude
            seed: Seed point
            mask: Segmentation mask
            visited: Visited tracker
            region_id: Region ID
            gradient_weight: Weight for gradient
        """
        height, width = image.shape
        seed_x, seed_y = seed

        if (seed_x < 0 or seed_x >= width or
            seed_y < 0 or seed_y >= height):
            return

        seed_intensity = float(image[seed_y, seed_x])
        seed_gradient = float(gradient[seed_y, seed_x])

        queue = deque([seed])
        visited[seed_y, seed_x] = True
        mask[seed_y, seed_x] = region_id

        while queue:
            x, y = queue.popleft()

            neighbors = self._get_neighbors(x, y, width, height)

            for nx, ny in neighbors:
                if not visited[ny, nx]:
                    neighbor_intensity = float(image[ny, nx])
                    neighbor_gradient = float(gradient[ny, nx])

                    # Combined similarity measure
                    intensity_diff = abs(neighbor_intensity - seed_intensity)
                    gradient_diff = abs(neighbor_gradient - seed_gradient)

                    combined_diff = (
                        (1 - gradient_weight) * intensity_diff +
                        gradient_weight * gradient_diff
                    )

                    if combined_diff <= self.threshold:
                        visited[ny, nx] = True
                        mask[ny, nx] = region_id
                        queue.append((nx, ny))


class SeededRegionGrowing:
    """
    Seeded region growing with multiple regions.
    """

    def __init__(self, threshold: float = 10.0):
        """
        Initialize seeded region grower.

        Args:
            threshold: Similarity threshold
        """
        self.threshold = threshold
        logging.info(f"Initialized seeded region growing: threshold={threshold}")

    def grow(self, image: np.ndarray,
            seed_mask: np.ndarray) -> np.ndarray:
        """
        Grow regions from seed mask.

        Args:
            image: Input grayscale image
            seed_mask: Mask with seed labels (0 = background, >0 = seed regions)

        Returns:
            Segmentation result
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape
        result = seed_mask.copy()
        visited = (seed_mask > 0)

        # Queue of (x, y, region_id)
        queue = deque()

        # Initialize queue with seed pixels
        for y in range(height):
            for x in range(width):
                if seed_mask[y, x] > 0:
                    queue.append((x, y, seed_mask[y, x]))

        # Grow regions
        while queue:
            x, y, region_id = queue.popleft()
            seed_intensity = float(gray[y, x])

            # Check 8-connected neighbors
            for dx, dy in [(-1, -1), (0, -1), (1, -1),
                          (-1, 0), (1, 0),
                          (-1, 1), (0, 1), (1, 1)]:
                nx, ny = x + dx, y + dy

                if (0 <= nx < width and 0 <= ny < height and
                    not visited[ny, nx]):

                    neighbor_intensity = float(gray[ny, nx])

                    if abs(neighbor_intensity - seed_intensity) <= self.threshold:
                        result[ny, nx] = region_id
                        visited[ny, nx] = True
                        queue.append((nx, ny, region_id))

        return result


class WatershedSegmentation:
    """
    Watershed segmentation wrapper.
    """

    @staticmethod
    def segment(image: np.ndarray, markers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply watershed segmentation.

        Args:
            image: Input image (BGR)
            markers: Optional marker image

        Returns:
            Segmentation result
        """
        # Ensure image is BGR
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()

        # If no markers provided, create automatic markers
        if markers is None:
            # Convert to grayscale
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                      kernel, iterations=2)

            # Sure background
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Sure foreground
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(),
                                      255, 0)

            # Unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Label markers
            _, markers = cv2.connectedComponents(sure_fg)

            # Add 1 to all labels (background is not 0 but 1)
            markers = markers + 1

            # Mark unknown regions as 0
            markers[unknown == 255] = 0

        # Apply watershed
        markers = markers.astype(np.int32)
        cv2.watershed(image_bgr, markers)

        return markers
