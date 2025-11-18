"""
Alert system module for intrusion detection.
Handles visual, audio, and logging alerts.
"""

import cv2
import numpy as np
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import platform


class AlertSystem:
    """
    Alert system class for managing intrusion alerts.
    """

    def __init__(self, visual: bool = True,
                 audio: bool = True,
                 log_file: str = "alerts.log",
                 save_screenshots: bool = True,
                 screenshot_dir: str = "data/output/screenshots"):
        """
        Initialize alert system.

        Args:
            visual: Enable visual alerts
            audio: Enable audio alerts
            log_file: Path to alert log file
            save_screenshots: Whether to save screenshots on alert
            screenshot_dir: Directory for saving screenshots
        """
        self.visual = visual
        self.audio = audio
        self.log_file = log_file
        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir

        # Create screenshot directory if needed
        if self.save_screenshots:
            os.makedirs(screenshot_dir, exist_ok=True)

        # Initialize alert log file
        self._init_log_file()

        # Alert counter
        self.alert_count = 0

        # Recent alerts tracking (to avoid spam)
        self.last_alert_time = {}
        self.alert_cooldown = 2.0  # seconds

        logging.info(f"Initialized alert system: visual={visual}, audio={audio}, "
                    f"log={log_file}")

    def _init_log_file(self) -> None:
        """Initialize the alert log file."""
        try:
            # Create log file directory if needed
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Write header if file doesn't exist
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w') as f:
                    f.write("Timestamp | ROI Name | Duration | Location | Screenshot\n")
                    f.write("-" * 80 + "\n")

            logging.info(f"Alert log initialized: {self.log_file}")

        except Exception as e:
            logging.error(f"Error initializing alert log: {e}")

    def trigger_alert(self, frame: np.ndarray,
                     intrusion_details: List[Dict[str, Any]],
                     frame_number: int = 0) -> np.ndarray:
        """
        Trigger alert for detected intrusions.

        Args:
            frame: Current video frame
            intrusion_details: List of intrusion information
            frame_number: Current frame number

        Returns:
            Frame with visual alerts applied
        """
        if not intrusion_details:
            return frame

        result_frame = frame.copy()
        current_time = datetime.now()

        for intrusion in intrusion_details:
            roi_name = intrusion['roi_name']

            # Check cooldown to avoid spam
            if self._check_cooldown(roi_name, current_time):
                continue

            # Update last alert time
            self.last_alert_time[roi_name] = current_time
            self.alert_count += 1

            # Visual alert
            if self.visual:
                result_frame = self._add_visual_alert(result_frame, intrusion)

            # Audio alert
            if self.audio:
                self._play_audio_alert()

            # Log alert
            self._log_alert(intrusion, current_time, frame_number)

            # Save screenshot
            if self.save_screenshots:
                self._save_screenshot(frame, intrusion, frame_number)

        return result_frame

    def _check_cooldown(self, roi_name: str, current_time: datetime) -> bool:
        """
        Check if alert is in cooldown period.

        Args:
            roi_name: ROI name
            current_time: Current timestamp

        Returns:
            True if in cooldown
        """
        if roi_name in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[roi_name]).total_seconds()
            return time_diff < self.alert_cooldown
        return False

    def _add_visual_alert(self, frame: np.ndarray,
                         intrusion: Dict[str, Any]) -> np.ndarray:
        """
        Add visual alert to frame.

        Args:
            frame: Input frame
            intrusion: Intrusion information

        Returns:
            Frame with visual alert
        """
        result = frame.copy()

        # Get bounding box
        x, y, w, h = intrusion['bbox']

        # Draw red bounding box (thick)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Add alert text on bounding box
        alert_text = f"INTRUSION: {intrusion['roi_name']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            alert_text, font, font_scale, thickness
        )

        # Draw background rectangle for text
        text_bg_y = max(y - text_height - 10, 0)
        cv2.rectangle(result,
                     (x, text_bg_y),
                     (x + text_width + 10, y),
                     (0, 0, 255), -1)

        # Draw text
        cv2.putText(result, alert_text,
                   (x + 5, y - 5),
                   font, font_scale, (255, 255, 255), thickness)

        # Add duration text
        duration_text = f"Time: {intrusion['duration']:.1f}s"
        cv2.putText(result, duration_text,
                   (x + 5, y + h + 20),
                   font, 0.5, (0, 0, 255), 2)

        # Add flashing "ALERT" banner at top
        height, width = result.shape[:2]
        banner_height = 60

        # Create semi-transparent red banner
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 255), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        # Add text to banner
        banner_text = "!!! INTRUSION DETECTED !!!"
        (banner_text_width, banner_text_height), _ = cv2.getTextSize(
            banner_text, cv2.FONT_HERSHEY_BOLD, 1.2, 3
        )
        text_x = (width - banner_text_width) // 2
        text_y = (banner_height + banner_text_height) // 2

        # Add text shadow
        cv2.putText(result, banner_text, (text_x + 2, text_y + 2),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (0, 0, 0), 5)
        # Add text
        cv2.putText(result, banner_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3)

        return result

    def _play_audio_alert(self) -> None:
        """
        Play audio alert (platform-dependent).
        """
        try:
            system = platform.system()

            if system == "Darwin":  # macOS
                os.system('afplay /System/Library/Sounds/Basso.aiff &')
            elif system == "Linux":
                os.system('aplay /usr/share/sounds/sound-icons/trumpet-12.wav &')
            elif system == "Windows":
                import winsound
                winsound.Beep(1000, 500)  # 1000 Hz for 500ms

        except Exception as e:
            logging.debug(f"Could not play audio alert: {e}")

    def _log_alert(self, intrusion: Dict[str, Any],
                   timestamp: datetime,
                   frame_number: int) -> None:
        """
        Log alert to file.

        Args:
            intrusion: Intrusion information
            timestamp: Alert timestamp
            frame_number: Frame number
        """
        try:
            log_entry = (
                f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{intrusion['roi_name']} | "
                f"{intrusion['duration']:.1f}s | "
                f"Frame {frame_number} | "
                f"Center: {intrusion['center']} | "
                f"Area: {intrusion['area']:.0f}px"
            )

            if self.save_screenshots:
                screenshot_name = f"alert_{self.alert_count:04d}.jpg"
                log_entry += f" | Screenshot: {screenshot_name}"

            with open(self.log_file, 'a') as f:
                f.write(log_entry + "\n")

            logging.info(f"Alert logged: {log_entry}")

        except Exception as e:
            logging.error(f"Error logging alert: {e}")

    def _save_screenshot(self, frame: np.ndarray,
                        intrusion: Dict[str, Any],
                        frame_number: int) -> None:
        """
        Save screenshot of alert.

        Args:
            frame: Frame to save
            intrusion: Intrusion information
            frame_number: Frame number
        """
        try:
            filename = f"alert_{self.alert_count:04d}.jpg"
            filepath = os.path.join(self.screenshot_dir, filename)

            # Add overlay information
            screenshot = frame.copy()
            info_text = [
                f"ROI: {intrusion['roi_name']}",
                f"Time: {intrusion['duration']:.1f}s",
                f"Frame: {frame_number}",
                f"Alert ID: {self.alert_count}"
            ]

            y_offset = 30
            for text in info_text:
                cv2.putText(screenshot, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25

            # Save screenshot
            cv2.imwrite(filepath, screenshot)
            logging.info(f"Screenshot saved: {filepath}")

        except Exception as e:
            logging.error(f"Error saving screenshot: {e}")

    def add_info_overlay(self, frame: np.ndarray,
                        frame_number: int,
                        fps: float = 0.0,
                        intrusion_count: int = 0) -> np.ndarray:
        """
        Add informational overlay to frame.

        Args:
            frame: Input frame
            frame_number: Current frame number
            fps: Current FPS
            intrusion_count: Current intrusion count

        Returns:
            Frame with overlay
        """
        result = frame.copy()
        height, width = result.shape[:2]

        # Create semi-transparent info panel
        panel_height = 80
        overlay = result.copy()
        cv2.rectangle(overlay, (0, height - panel_height),
                     (300, height), (50, 50, 50), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        # Add information text
        info_texts = [
            f"Frame: {frame_number}",
            f"FPS: {fps:.1f}",
            f"Alerts: {self.alert_count}",
            f"Active: {intrusion_count}"
        ]

        y_start = height - panel_height + 20
        for i, text in enumerate(info_texts):
            y_pos = y_start + i * 20
            cv2.putText(result, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, timestamp, (width - 200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return result

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alerts.

        Returns:
            Alert summary dictionary
        """
        return {
            'total_alerts': self.alert_count,
            'log_file': self.log_file,
            'screenshot_dir': self.screenshot_dir if self.save_screenshots else None
        }

    def reset(self) -> None:
        """
        Reset alert system counters.
        """
        self.alert_count = 0
        self.last_alert_time.clear()
        logging.info("Alert system reset")


class AlertNotifier:
    """
    Extended alert notifier for external notifications.
    (Email, SMS, webhooks, etc.)
    """

    def __init__(self):
        """Initialize alert notifier."""
        self.notification_enabled = False
        logging.info("Alert notifier initialized")

    def send_email_alert(self, subject: str, body: str,
                        attachment: Optional[str] = None) -> bool:
        """
        Send email alert (placeholder - requires email config).

        Args:
            subject: Email subject
            body: Email body
            attachment: Optional attachment path

        Returns:
            Success status
        """
        # TODO: Implement email sending
        logging.info(f"Email alert: {subject}")
        return False

    def send_webhook_notification(self, url: str, data: Dict[str, Any]) -> bool:
        """
        Send webhook notification (placeholder).

        Args:
            url: Webhook URL
            data: Data to send

        Returns:
            Success status
        """
        # TODO: Implement webhook
        logging.info(f"Webhook notification to {url}")
        return False
