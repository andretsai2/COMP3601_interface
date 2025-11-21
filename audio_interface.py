"""
Audio Interface
Monitors filesystem for new audio recordings and queues them for processing
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
import queue
import threading

logger = logging.getLogger(__name__)


class AudioInterface:
    """
    Monitors directory for new WAV files from the C audio recorder
    """

    def __init__(self, recordings_dir: str, poll_interval: float = 1.0):
        """
        Initialize audio interface

        Args:
            recordings_dir: Directory to monitor for new recordings
            poll_interval: How often to check for new files (seconds)
        """
        self.recordings_dir = Path(recordings_dir)
        self.poll_interval = poll_interval
        self.audio_queue = queue.Queue(maxsize=10)
        self.processed_files = set()
        self.running = False
        self.monitor_thread = None

        # Create directory if it doesn't exist
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Audio Interface initialized")
        logger.info(f"  Monitoring: {self.recordings_dir}")
        logger.info(f"  Poll interval: {self.poll_interval}s")

    def start(self):
        """Start monitoring for new files"""
        if self.running:
            logger.warning("Already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started file monitoring")

    def stop(self):
        """Stop monitoring"""
        if not self.running:
            return

        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped file monitoring")

    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        logger.info("Monitoring thread started")

        while self.running:
            try:
                # Scan for new WAV files
                self._scan_for_new_files()

                # Sleep before next scan
                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)  # Wait before retrying

        logger.info("Monitoring thread stopped")

    def _scan_for_new_files(self):
        """Scan directory for new WAV files"""
        try:
            if not self.recordings_dir.exists():
                return

            # Find all WAV files
            wav_files = list(self.recordings_dir.glob("*.wav"))
            wav_files += list(self.recordings_dir.glob("*.WAV"))

            for wav_file in wav_files:
                file_path = str(wav_file)

                # Skip if already processed
                if file_path in self.processed_files:
                    continue

                # Check if file is ready (not being written)
                if not self._is_file_ready(wav_file):
                    continue

                # Mark as processed
                self.processed_files.add(file_path)

                # Get file info
                file_size = wav_file.stat().st_size / (1024**2)  # MB

                # Queue for processing
                try:
                    audio_info = {
                        'file_path': file_path,
                        'file_name': wav_file.name,
                        'file_size_mb': round(file_size, 2),
                        'timestamp': time.time()
                    }

                    self.audio_queue.put(audio_info, block=False)
                    logger.info(f"[NEW FILE] {wav_file.name} ({file_size:.2f}MB)")

                except queue.Full:
                    logger.warning(f"Queue full, skipping: {wav_file.name}")
                    self.processed_files.remove(file_path)  # Allow retry

        except Exception as e:
            logger.error(f"Error scanning for files: {e}")

    def _is_file_ready(self, file_path: Path) -> bool:
        """
        Check if file is completely written and ready to process

        Args:
            file_path: Path to file

        Returns:
            True if file is ready
        """
        try:
            # Check if file exists and has non-zero size
            if not file_path.exists():
                return False

            file_size = file_path.stat().st_size
            if file_size == 0:
                return False

            # Check if file is still being written
            # Wait a moment and check if size changes
            initial_size = file_size
            time.sleep(0.2)

            if not file_path.exists():
                return False

            final_size = file_path.stat().st_size

            # If size changed, file is still being written
            if initial_size != final_size:
                return False

            # Try to open file (will fail if locked by writer)
            try:
                with open(file_path, 'rb') as f:
                    # Try to read WAV header
                    header = f.read(44)
                    if len(header) < 44:
                        return False

                    # Check for RIFF header
                    if header[0:4] != b'RIFF':
                        logger.warning(f"Invalid WAV header: {file_path.name}")
                        return False
            except (IOError, PermissionError):
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking file readiness: {e}")
            return False

    def get_audio(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get next audio file from queue

        Args:
            timeout: Timeout in seconds

        Returns:
            Audio info dict or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def scan_existing_files(self) -> int:
        """
        Scan for existing WAV files on startup

        Returns:
            Number of files found
        """
        if not self.recordings_dir.exists():
            logger.warning(f"Recordings directory does not exist: {self.recordings_dir}")
            return 0

        wav_files = list(self.recordings_dir.glob("*.wav"))
        wav_files += list(self.recordings_dir.glob("*.WAV"))

        logger.info(f"Found {len(wav_files)} existing WAV files")

        # Queue them all
        queued = 0
        for wav_file in wav_files:
            file_path = str(wav_file)

            if file_path in self.processed_files:
                continue

            if self._is_file_ready(wav_file):
                file_size = wav_file.stat().st_size / (1024**2)

                audio_info = {
                    'file_path': file_path,
                    'file_name': wav_file.name,
                    'file_size_mb': round(file_size, 2),
                    'timestamp': time.time()
                }

                try:
                    self.audio_queue.put(audio_info, block=False)
                    self.processed_files.add(file_path)
                    queued += 1
                except queue.Full:
                    break

        logger.info(f"Queued {queued} existing files for processing")
        return queued

    def get_stats(self) -> Dict:
        """
        Get interface statistics

        Returns:
            Stats dictionary
        """
        return {
            'running': self.running,
            'recordings_dir': str(self.recordings_dir),
            'processed_count': len(self.processed_files),
            'queue_size': self.audio_queue.qsize(),
            'queue_max': self.audio_queue.maxsize
        }