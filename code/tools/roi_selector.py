"""
Interactive ROI (Region of Interest) selector tool.
Allows users to define restricted areas by clicking on video frames.
"""

import cv2
import json
import argparse
import logging
import numpy as np
from typing import List, Tuple


class ROISelector:
    """
    Interactive ROI selector for defining restricted areas.
    """

    def __init__(self, video_path: str):
        """
        Initialize ROI selector.

        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.points = []
        self.rois = []
        self.frame = None
        self.display_frame = None
        self.current_roi_name = "Area 1"
        self.window_name = "ROI Selector"

        logging.basicConfig(level=logging.INFO)
        logging.info(f"Initialized ROI selector for: {video_path}")

    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events.

        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: Add point
            self.points.append([x, y])
            logging.info(f"Point added: ({x}, {y})")
            self._update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: Finish current ROI
            if len(self.points) >= 3:
                roi = {
                    "name": self.current_roi_name,
                    "type": "polygon",
                    "points": self.points.copy(),
                    "color": self._generate_color(len(self.rois))
                }
                self.rois.append(roi)
                logging.info(f"ROI '{self.current_roi_name}' saved with {len(self.points)} points")

                # Prepare for next ROI
                self.points = []
                self.current_roi_name = f"Area {len(self.rois) + 1}"
                self._update_display()
            else:
                logging.warning("Need at least 3 points to create ROI")

    def _generate_color(self, index: int) -> List[int]:
        """
        Generate unique color for ROI.

        Args:
            index: ROI index

        Returns:
            BGR color list
        """
        colors = [
            [255, 0, 0],      # Blue
            [0, 0, 255],      # Red
            [0, 255, 0],      # Green
            [255, 255, 0],    # Cyan
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Yellow
        ]
        return colors[index % len(colors)]

    def _update_display(self):
        """Update the display with current points and ROIs."""
        if self.frame is None:
            return

        self.display_frame = self.frame.copy()

        # Draw existing ROIs
        for roi in self.rois:
            pts = np.array(roi["points"], dtype=np.int32)
            cv2.polylines(self.display_frame, [pts], True,
                        tuple(roi["color"]), 2)

            # Draw ROI name
            center = pts.mean(axis=0).astype(int)
            cv2.putText(self.display_frame, roi["name"], tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(roi["color"]), 2)

        # Draw current points
        for i, pt in enumerate(self.points):
            cv2.circle(self.display_frame, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(self.display_frame, str(i + 1),
                       (pt[0] + 10, pt[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw lines between current points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(self.display_frame, tuple(self.points[i]),
                        tuple(self.points[i + 1]), (0, 255, 0), 2)

        # Draw instructions
        self._draw_instructions()

        # Show frame
        cv2.imshow(self.window_name, self.display_frame)

    def _draw_instructions(self):
        """Draw instruction panel on display."""
        instructions = [
            "LEFT CLICK: Add point",
            "RIGHT CLICK: Finish ROI",
            "'c': Clear current points",
            "'d': Delete last ROI",
            "'s': Save and exit",
            "'q': Quit without saving"
        ]

        # Create semi-transparent panel
        panel_height = len(instructions) * 25 + 20
        panel = np.zeros((panel_height, 350, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)

        # Add text
        for i, text in enumerate(instructions):
            cv2.putText(panel, text, (10, 20 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Overlay panel on frame
        h, w = self.display_frame.shape[:2]
        self.display_frame[10:10 + panel_height, 10:360] = \
            cv2.addWeighted(self.display_frame[10:10 + panel_height, 10:360],
                          0.3, panel, 0.7, 0)

        # Add current ROI info
        info_text = f"Current ROI: {self.current_roi_name} ({len(self.points)} points)"
        cv2.putText(self.display_frame, info_text,
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 255), 2)

        total_text = f"Total ROIs: {len(self.rois)}"
        cv2.putText(self.display_frame, total_text,
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 255), 2)

    def run(self):
        """Run the ROI selector interface."""
        # Open video
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            logging.error("Failed to open video file")
            return

        # Read first frame
        ret, self.frame = cap.read()
        cap.release()

        if not ret:
            logging.error("Failed to read video frame")
            return

        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        logging.info("ROI Selector started")
        logging.info("Left click to add points, right click to finish ROI")

        # Initial display
        self._update_display()

        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Save and exit
                self.save_rois()
                break

            elif key == ord('q'):
                # Quit without saving
                logging.info("Exiting without saving")
                break

            elif key == ord('c'):
                # Clear current points
                self.points = []
                logging.info("Current points cleared")
                self._update_display()

            elif key == ord('d'):
                # Delete last ROI
                if self.rois:
                    deleted = self.rois.pop()
                    logging.info(f"Deleted ROI: {deleted['name']}")
                    self.current_roi_name = f"Area {len(self.rois) + 1}"
                    self._update_display()
                else:
                    logging.warning("No ROIs to delete")

        cv2.destroyAllWindows()

    def save_rois(self, output_path: str = "data/roi/restricted_area.json"):
        """
        Save ROIs to JSON file.

        Args:
            output_path: Output file path
        """
        if not self.rois:
            logging.warning("No ROIs to save")
            return

        output = {"restricted_areas": self.rois}

        try:
            # Create directory if needed
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

            logging.info(f"Saved {len(self.rois)} ROI(s) to {output_path}")

            # Print summary
            print("\n" + "=" * 60)
            print("ROI SUMMARY")
            print("=" * 60)
            for i, roi in enumerate(self.rois):
                print(f"\nROI {i + 1}: {roi['name']}")
                print(f"  Type: {roi['type']}")
                print(f"  Points: {len(roi['points'])}")
                print(f"  Color: {roi['color']}")
            print("=" * 60)

        except Exception as e:
            logging.error(f"Error saving ROIs: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive ROI Selector for Intrusion Detection"
    )

    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to video file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/roi/restricted_area.json',
        help='Output JSON file path'
    )

    args = parser.parse_args()

    # Create and run selector
    selector = ROISelector(args.video)
    selector.run()
    selector.save_rois(args.output)


if __name__ == "__main__":
    main()
