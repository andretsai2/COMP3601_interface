"""
Storage Manager
Handles disk space monitoring, file cleanup, and storage optimization
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages storage: deletes originals, monitors space, removes old clips
    """

    def __init__(self,
                 min_free_space_gb: float = 2.0,
                 max_clips: int = 10000):
        """
        Initialize storage manager

        Args:
            min_free_space_gb: Minimum free space before cleanup (GB)
            max_clips: Maximum number of detection clips to keep
        """
        self.min_free_space = min_free_space_gb * 1024**3  # Convert to bytes
        self.max_clips = max_clips

        logger.info(f"Storage Manager initialized:")
        logger.info(f"  Min free space: {min_free_space_gb}GB")
        logger.info(f"  Max clips: {max_clips}")

    def get_free_space(self, path: str = "/") -> int:
        """
        Get free space in bytes for given path

        Args:
            path: Directory path to check

        Returns:
            Free space in bytes
        """
        try:
            stat = os.statvfs(path)
            free_bytes = stat.f_bavail * stat.f_frsize
            return free_bytes
        except Exception as e:
            logger.error(f"Failed to get free space: {e}")
            return 0

    def get_free_space_gb(self, path: str = "/") -> float:
        """Get free space in GB"""
        return self.get_free_space(path) / (1024**3)

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file safely

        Args:
            file_path: Path to file to delete

        Returns:
            True if deleted successfully
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                size_mb = 0  # Already deleted
                logger.info(f"[DELETE] {Path(file_path).name}")
                return True
            else:
                logger.warning(f"File not found: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False

    def cleanup_original(self, wav_file: str, had_detections: bool = False) -> bool:
        """
        Delete original WAV file after processing

        Args:
            wav_file: Path to original recording
            had_detections: Whether detections were found

        Returns:
            True if deleted
        """
        try:
            file_size = os.path.getsize(wav_file) / (1024**2)  # MB

            if self.delete_file(wav_file):
                if had_detections:
                    logger.info(f"  Deleted original (kept clips): {Path(wav_file).name} ({file_size:.1f}MB)")
                else:
                    logger.info(f"  Deleted original (no detections): {Path(wav_file).name} ({file_size:.1f}MB)")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to cleanup {wav_file}: {e}")
            return False

    def cleanup_old_clips(self, clips_dir: str) -> int:
        """
        Remove oldest clips if exceeding max count

        Args:
            clips_dir: Directory containing detection clips

        Returns:
            Number of clips deleted
        """
        try:
            clips_path = Path(clips_dir)
            if not clips_path.exists():
                return 0

            # Get all WAV clips sorted by creation time
            clips = sorted(clips_path.glob("*.wav"), key=os.path.getctime)

            num_clips = len(clips)
            if num_clips <= self.max_clips:
                return 0

            # Delete oldest clips
            num_to_delete = num_clips - self.max_clips
            deleted = 0

            for clip in clips[:num_to_delete]:
                try:
                    size = clip.stat().st_size / 1024  # KB
                    os.remove(clip)
                    logger.info(f"[DELETE] Removed old clip: {clip.name} ({size:.1f}KB)")
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete {clip}: {e}")

            logger.info(f"Cleaned up {deleted} old clips (keeping newest {self.max_clips})")
            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old clips: {e}")
            return 0

    def ensure_free_space(self, clips_dir: str) -> bool:
        """
        Ensure sufficient free space by deleting old clips if needed

        Args:
            clips_dir: Directory containing clips

        Returns:
            True if space is sufficient
        """
        free_gb = self.get_free_space_gb()

        if free_gb >= (self.min_free_space / 1024**3):
            return True

        logger.warning(f"[WARNING] Low disk space: {free_gb:.2f}GB free")

        try:
            clips_path = Path(clips_dir)
            if not clips_path.exists():
                return False

            # Get clips sorted by age (oldest first)
            clips = sorted(clips_path.glob("*.wav"), key=os.path.getctime)

            deleted = 0
            for clip in clips:
                # Check if we have enough space now
                if self.get_free_space_gb() >= (self.min_free_space / 1024**3):
                    break

                try:
                    size = clip.stat().st_size
                    os.remove(clip)
                    deleted += 1
                    logger.info(f"[DELETE] Freed {size / 1024:.1f}KB: {clip.name}")
                except Exception as e:
                    logger.error(f"Failed to delete {clip}: {e}")

            final_free = self.get_free_space_gb()
            logger.info(f"Deleted {deleted} clips. Free space: {final_free:.2f}GB")

            return final_free >= (self.min_free_space / 1024**3)

        except Exception as e:
            logger.error(f"Failed to ensure free space: {e}")
            return False

    def get_directory_size(self, directory: str) -> float:
        """
        Get total size of directory in MB

        Args:
            directory: Path to directory

        Returns:
            Size in MB
        """
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024**2)
        except Exception as e:
            logger.error(f"Failed to get directory size: {e}")
            return 0.0

    def get_storage_stats(self, recordings_dir: str, clips_dir: str) -> Dict:
        """
        Get comprehensive storage statistics

        Args:
            recordings_dir: Directory with original recordings
            clips_dir: Directory with detection clips

        Returns:
            Dictionary with storage stats
        """
        try:
            free_space = self.get_free_space_gb()

            # Count files
            rec_path = Path(recordings_dir)
            clips_path = Path(clips_dir)

            rec_files = list(rec_path.glob("*.wav")) if rec_path.exists() else []
            clip_files = list(clips_path.glob("*.wav")) if clips_path.exists() else []

            rec_size = self.get_directory_size(recordings_dir) if rec_path.exists() else 0
            clip_size = self.get_directory_size(clips_dir) if clips_path.exists() else 0

            stats = {
                'free_space_gb': round(free_space, 2),
                'recordings_count': len(rec_files),
                'recordings_size_mb': round(rec_size, 2),
                'clips_count': len(clip_files),
                'clips_size_mb': round(clip_size, 2),
                'total_used_mb': round(rec_size + clip_size, 2),
                'space_ok': free_space >= (self.min_free_space / 1024**3)
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                'free_space_gb': 0,
                'recordings_count': 0,
                'recordings_size_mb': 0,
                'clips_count': 0,
                'clips_size_mb': 0,
                'total_used_mb': 0,
                'space_ok': False
            }

    def print_storage_status(self, recordings_dir: str, clips_dir: str):
        """
        Print storage status to log

        Args:
            recordings_dir: Recordings directory
            clips_dir: Clips directory
        """
        stats = self.get_storage_stats(recordings_dir, clips_dir)

        logger.info("=" * 50)
        logger.info("STORAGE STATUS")
        logger.info("=" * 50)
        logger.info(f"Free Space: {stats['free_space_gb']}GB")
        logger.info(f"Recordings: {stats['recordings_count']} files ({stats['recordings_size_mb']}MB)")
        logger.info(f"Clips: {stats['clips_count']} files ({stats['clips_size_mb']}MB)")
        logger.info(f"Total Used: {stats['total_used_mb']}MB")
        logger.info(f"Status: {'OK' if stats['space_ok'] else 'LOW SPACE'}")
        logger.info("=" * 50)