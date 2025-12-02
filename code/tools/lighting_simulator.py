"""
Lighting Condition Simulator Tool.
Creates video variants with different lighting conditions (low-light, night)
while maintaining enough visibility for the intrusion detection system.
"""

import cv2
import numpy as np
import argparse
import logging
import os
from typing import Tuple, Optional
from pathlib import Path


class LightingSimulator:
    """
    Simulates different lighting conditions on video input.
    """

    # Lighting presets
    PRESETS = {
        'normal': {
            'brightness_factor': 1.0,
            'noise_level': 0,
            'night_vision': False,
            'description': 'Normal daylight conditions'
        },
        'low-light': {
            'brightness_factor': 0.45,
            'noise_level': 8,
            'night_vision': False,
            'description': 'Low-light conditions (dusk, cloudy, indoor)'
        },
        'night': {
            'brightness_factor': 0.20,
            'noise_level': 15,
            'night_vision': True,
            'description': 'Night conditions with night-vision effect'
        }
    }

    def __init__(self, input_video: str, output_video: str,
                 lighting_mode: str = 'low-light',
                 custom_brightness: Optional[float] = None,
                 custom_noise: Optional[int] = None):
        """
        Initialize lighting simulator.

        Args:
            input_video: Path to input video
            output_video: Path to output video
            lighting_mode: Preset mode ('normal', 'low-light', 'night')
            custom_brightness: Custom brightness factor (overrides preset)
            custom_noise: Custom noise level (overrides preset)
        """
        self.input_video = input_video
        self.output_video = output_video
        self.mode = lighting_mode

        if lighting_mode not in self.PRESETS:
            raise ValueError(f"Invalid mode. Choose from: {list(self.PRESETS.keys())}")

        # Get preset parameters
        preset = self.PRESETS[lighting_mode]
        self.brightness_factor = custom_brightness if custom_brightness else preset['brightness_factor']
        self.noise_level = custom_noise if custom_noise else preset['noise_level']
        self.night_vision = preset['night_vision']

        logging.basicConfig(level=logging.INFO)
        logging.info(f"Lighting Simulator initialized")
        logging.info(f"Mode: {lighting_mode} - {preset['description']}")
        logging.info(f"Brightness: {self.brightness_factor:.2f}")
        logging.info(f"Noise level: {self.noise_level}")
        logging.info(f"Night vision: {self.night_vision}")

    def add_gaussian_noise(self, frame: np.ndarray, noise_level: int) -> np.ndarray:
        """
        Add Gaussian noise to simulate low-light sensor noise.

        Args:
            frame: Input frame
            noise_level: Standard deviation of noise

        Returns:
            Noisy frame
        """
        if noise_level == 0:
            return frame

        noise = np.random.randn(*frame.shape) * noise_level
        noisy_frame = frame.astype(np.float32) + noise
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        return noisy_frame

    def adjust_brightness(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust frame brightness.

        Args:
            frame: Input frame
            factor: Brightness multiplication factor

        Returns:
            Adjusted frame
        """
        adjusted = frame.astype(np.float32) * factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted

    def apply_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply night vision effect (green tint, enhanced contrast).

        Args:
            frame: Input frame

        Returns:
            Night vision frame
        """
        # Convert to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Create green tint (night vision look)
        night_vision = np.zeros_like(frame)
        night_vision[:, :, 1] = enhanced  # Green channel
        night_vision[:, :, 0] = enhanced // 4  # Slight blue

        return night_vision

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with lighting effects.

        Args:
            frame: Input frame

        Returns:
            Processed frame
        """
        # Apply brightness adjustment
        processed = self.adjust_brightness(frame, self.brightness_factor)

        # Add noise
        processed = self.add_gaussian_noise(processed, self.noise_level)

        # Apply night vision if enabled
        if self.night_vision:
            processed = self.apply_night_vision(processed)

        return processed

    def process_video(self) -> bool:
        """
        Process entire video and save result.

        Returns:
            True if successful, False otherwise
        """
        # Open input video
        cap = cv2.VideoCapture(self.input_video)

        if not cap.isOpened():
            logging.error(f"Failed to open video: {self.input_video}")
            return False

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Create output directory if needed
        os.makedirs(os.path.dirname(self.output_video), exist_ok=True)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video, fourcc, fps, (width, height))

        if not out.isOpened():
            logging.error(f"Failed to create output video: {self.output_video}")
            cap.release()
            return False

        logging.info("Processing video...")

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Process frame
            processed_frame = self.process_frame(frame)

            # Write frame
            out.write(processed_frame)

            frame_count += 1

            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logging.info(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")

        # Cleanup
        cap.release()
        out.release()

        logging.info(f"Processing complete! Output saved to: {self.output_video}")
        logging.info(f"Processed {frame_count} frames")

        return True

    def create_comparison_preview(self, output_path: str = None) -> bool:
        """
        Create a side-by-side comparison image of original vs processed.

        Args:
            output_path: Path to save comparison image

        Returns:
            True if successful
        """
        # Open both videos
        cap_orig = cv2.VideoCapture(self.input_video)
        cap_proc = cv2.VideoCapture(self.output_video)

        if not cap_orig.isOpened() or not cap_proc.isOpened():
            logging.error("Failed to open videos for comparison")
            return False

        # Read first frames
        ret1, frame_orig = cap_orig.read()
        ret2, frame_proc = cap_proc.read()

        cap_orig.release()
        cap_proc.release()

        if not ret1 or not ret2:
            logging.error("Failed to read frames for comparison")
            return False

        # Create side-by-side comparison
        h, w = frame_orig.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = frame_orig
        comparison[:, w:] = frame_proc

        # Add labels
        cv2.putText(comparison, "Original", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(comparison, f"{self.mode.upper()}", (w + 20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Save comparison
        if output_path is None:
            base_name = os.path.splitext(self.output_video)[0]
            output_path = f"{base_name}_comparison.jpg"

        cv2.imwrite(output_path, comparison)
        logging.info(f"Comparison preview saved to: {output_path}")

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Lighting Condition Simulator for Intrusion Detection Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create low-light version of input-01
  python lighting_simulator.py --input data/input/input-01.mp4 --mode low-light

  # Create night version with custom parameters
  python lighting_simulator.py --input data/input/input-01.mp4 --mode night --brightness 0.15

  # Create custom lighting condition
  python lighting_simulator.py --input data/input/input-01.mp4 --mode low-light --brightness 0.3 --noise 12

Preset modes:
  normal     : Normal daylight (brightness=1.0, noise=0)
  low-light  : Low-light conditions (brightness=0.45, noise=8)
  night      : Night with night-vision (brightness=0.20, noise=15)
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input video file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output video (default: auto-generated based on mode)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['normal', 'low-light', 'night'],
        default='low-light',
        help='Lighting preset mode (default: low-light)'
    )

    parser.add_argument(
        '--brightness',
        type=float,
        default=None,
        help='Custom brightness factor (0.0-1.0, overrides preset)'
    )

    parser.add_argument(
        '--noise',
        type=int,
        default=None,
        help='Custom noise level (0-30, overrides preset)'
    )

    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Create comparison preview image'
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output is None:
        input_path = Path(args.input)
        input_name = input_path.stem
        output_name = f"{input_name}-{args.mode}{input_path.suffix}"
        args.output = str(input_path.parent / output_name)

    # Create simulator
    simulator = LightingSimulator(
        input_video=args.input,
        output_video=args.output,
        lighting_mode=args.mode,
        custom_brightness=args.brightness,
        custom_noise=args.noise
    )

    # Process video
    success = simulator.process_video()

    if success and args.comparison:
        simulator.create_comparison_preview()

    if success:
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Mode:   {args.mode}")
        print(f"Brightness: {simulator.brightness_factor:.2f}")
        print(f"Noise: {simulator.noise_level}")
        print("=" * 70)


if __name__ == "__main__":
    main()
