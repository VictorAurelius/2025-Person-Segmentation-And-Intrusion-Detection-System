"""
Main application for intrusion detection system.
"""

import cv2
import numpy as np
import argparse
import logging
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (load_config, load_roi_definitions, draw_roi,
                   draw_bounding_boxes, add_text_overlay, setup_logging)
from motion_detector import MotionDetector
from adaptive_threshold import AdaptiveThreshold, CLAHEProcessor
from edge_detector import EdgeDetector
from intrusion_detector import IntrusionDetector
from alert_system import AlertSystem


class IntrusionDetectionSystem:
    """
    Main intrusion detection system class.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the intrusion detection system.

        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        setup_logging(logging.INFO)
        logging.info("=" * 80)
        logging.info("Initializing Intrusion Detection System")
        logging.info("=" * 80)

        # Load configuration
        try:
            self.config = load_config(config_path)
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            sys.exit(1)

        # Initialize components
        self._init_components()

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        logging.info("System initialized successfully")

    def _init_components(self) -> None:
        """Initialize all system components."""
        # Motion detector
        motion_config = self.config['motion']
        self.motion_detector = MotionDetector(
            method=motion_config['method'],
            history=motion_config['history'],
            threshold=motion_config['threshold'],
            detect_shadows=motion_config['detect_shadows']
        )

        # Adaptive threshold
        threshold_config = self.config['threshold']
        self.adaptive_threshold = AdaptiveThreshold(
            method=threshold_config['method'],
            block_size=threshold_config['block_size'],
            C=threshold_config['C']
        )

        # CLAHE processor for low-light enhancement
        self.clahe = CLAHEProcessor(clip_limit=2.0, tile_grid_size=(8, 8))

        # Edge detector
        edge_config = self.config['edge']
        self.edge_detector = EdgeDetector(
            method=edge_config['method'],
            low_threshold=edge_config['low_threshold'],
            high_threshold=edge_config['high_threshold']
        )

        # Load ROI definitions
        intrusion_config = self.config['intrusion']
        try:
            roi_definitions = load_roi_definitions(intrusion_config['roi_file'])
        except Exception as e:
            logging.error(f"Failed to load ROI definitions: {e}")
            logging.warning("Using empty ROI list")
            roi_definitions = []

        # Intrusion detector
        self.intrusion_detector = IntrusionDetector(
            roi_definitions=roi_definitions,
            overlap_threshold=intrusion_config['overlap_threshold'],
            time_threshold=intrusion_config['time_threshold'],
            min_object_area=intrusion_config['min_object_area']
        )

        # Alert system
        alert_config = self.config['alert']
        self.alert_system = AlertSystem(
            visual=alert_config['visual'],
            audio=alert_config['audio'],
            log_file=alert_config['log_file'],
            save_screenshots=alert_config['save_screenshots']
        )

        # Store ROI definitions for drawing
        self.roi_definitions = roi_definitions

    def process_video(self, source: str = None, output_path: str = None) -> None:
        """
        Process video for intrusion detection.

        Args:
            source: Video source (file path or camera index)
            output_path: Output video path
        """
        # Use config defaults if not specified
        if source is None:
            source = self.config['video']['source']
        if output_path is None:
            output_path = self.config['output']['output_path']

        # Open video source
        if isinstance(source, str) and not source.isdigit():
            cap = cv2.VideoCapture(source)
            logging.info(f"Processing video file: {source}")
        else:
            cap = cv2.VideoCapture(int(source) if isinstance(source, str) else source)
            logging.info(f"Processing camera source: {source}")

        if not cap.isOpened():
            logging.error("Failed to open video source")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Video properties: {width}x{height} @ {fps:.2f} FPS")
        if total_frames > 0:
            logging.info(f"Total frames: {total_frames}")

        # Setup video writer
        video_writer = None
        if self.config['output']['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logging.info(f"Saving output to: {output_path}")

        # Processing loop
        self.frame_count = 0
        self.start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Process frame
                result_frame = self._process_frame(frame)

                # Add info overlay
                result_frame = self.alert_system.add_info_overlay(
                    result_frame,
                    self.frame_count,
                    self.fps
                )

                # Save to output video
                if video_writer is not None:
                    video_writer.write(result_frame)

                # Display frame
                if self.config['output']['show_realtime']:
                    cv2.imshow("Intrusion Detection System", result_frame)

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logging.info("User requested quit")
                        break
                    elif key == ord('p'):
                        logging.info("Paused - press any key to continue")
                        cv2.waitKey(0)
                    elif key == ord('r'):
                        self.motion_detector.reset_background_model()
                        logging.info("Background model reset")

                # Update FPS
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed

                # Log progress
                if self.frame_count % 100 == 0:
                    if total_frames > 0:
                        progress = (self.frame_count / total_frames) * 100
                        logging.info(f"Progress: {self.frame_count}/{total_frames} "
                                   f"({progress:.1f}%) - FPS: {self.fps:.1f}")
                    else:
                        logging.info(f"Processed {self.frame_count} frames - "
                                   f"FPS: {self.fps:.1f}")

        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")

        finally:
            # Cleanup
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()

            # Print summary
            self._print_summary()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the detection pipeline.

        Args:
            frame: Input frame

        Returns:
            Processed frame with visualizations
        """
        # Apply CLAHE for low-light enhancement (optional)
        # frame = self.clahe.apply(frame)

        # Step 1: Motion detection
        fg_mask, processed_mask = self.motion_detector.detect(frame)

        # Step 2: Find contours
        contours = self.motion_detector.get_contours(
            processed_mask,
            min_area=self.config['intrusion']['min_object_area']
        )

        # Step 3: Intrusion detection
        current_time = time.time()
        intrusion_flags, intrusion_details = self.intrusion_detector.detect_intrusions(
            contours, current_time
        )

        # Step 4: Create visualization
        result = frame.copy()

        # Draw ROI areas
        result = draw_roi(result, self.roi_definitions)

        # Draw bounding boxes
        result = draw_bounding_boxes(
            result,
            contours,
            intrusion_flags,
            min_area=self.config['intrusion']['min_object_area']
        )

        # Step 5: Trigger alerts
        if intrusion_details:
            result = self.alert_system.trigger_alert(
                result,
                intrusion_details,
                self.frame_count
            )

        return result

    def _print_summary(self) -> None:
        """Print processing summary."""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

        logging.info("=" * 80)
        logging.info("PROCESSING SUMMARY")
        logging.info("=" * 80)
        logging.info(f"Total frames processed: {self.frame_count}")
        logging.info(f"Total time: {elapsed:.2f} seconds")
        logging.info(f"Average FPS: {avg_fps:.2f}")

        alert_summary = self.alert_system.get_alert_summary()
        logging.info(f"Total alerts: {alert_summary['total_alerts']}")
        logging.info(f"Alert log: {alert_summary['log_file']}")

        if alert_summary['screenshot_dir']:
            logging.info(f"Screenshots: {alert_summary['screenshot_dir']}")

        logging.info("=" * 80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Person Segmentation & Intrusion Detection System"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Video source (file path or camera index)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path'
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable real-time display'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    try:
        # Create system
        system = IntrusionDetectionSystem(args.config)

        # Override display setting if specified
        if args.no_display:
            system.config['output']['show_realtime'] = False

        # Process video
        system.process_video(
            source=args.source,
            output_path=args.output
        )

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
