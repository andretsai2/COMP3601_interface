
import os
import sys
import json
import logging
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Import our modules
from src.base_model import BaseWildlifeModel
from src.placeholder_model import PlaceholderModel
from src.yamnet_keras_model import YAMNetKerasModel
from src.yamnet_tflite_model import YAMNetTFLiteModel
from src.audio_interface import AudioInterface
from src.storage_manager import StorageManager


class WildlifeDetector:
    """
    Main wildlife detection system coordinator
    """

    def __init__(self, config_path: str = "detector_config.json"):
        """
        Initialize wildlife detector

        Args:
            config_path: Path to configuration file
        """
        print("=" * 60)
        print("Wildlife Detection System Starting")
        print("=" * 60)

        # Load configuration
        self.config = self.load_config(config_path)

        # Setup logging
        self.setup_logging()

        logging.info("Initializing Wildlife Detector...")

        # Create directories
        self.create_directories()

        # Initialize model
        self.model = self.load_model()

        # Initialize audio interface
        self.audio_interface = AudioInterface(
            recordings_dir=self.config['paths']['recordings'],
            poll_interval=1.0
        )

        # Initialize storage manager
        self.storage_manager = StorageManager(
            min_free_space_gb=self.config['storage']['min_free_gb'],
            max_clips=self.config['storage']['max_clips']
        )

        # Load species targets
        self.species_targets = self.load_species_targets()

        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_detections': 0,
            'species_counts': {},
            'start_time': datetime.now().isoformat()
        }

        # Running flag
        self.running = False

        logging.info(f"Monitoring: {self.config['paths']['recordings']}")
        logging.info(f"Model: {self.config['model']['name']}")
        logging.info(f"Target species: {len(self.species_targets.get('species', []))}")
        logging.info(f"Clips directory: {self.config['paths']['clips']}")

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in config: {e}")
            sys.exit(1)

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "system.log"
        log_level = getattr(logging, self.config['logging']['level'])

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['paths']['recordings'],
            self.config['paths']['clips'],
            self.config['paths']['logs'],
            self.config['paths']['models']
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logging.info(f"Directory ready: {dir_path}")

    def load_model(self) -> BaseWildlifeModel:
        """Load ML model based on configuration"""
        model_type = self.config['model']['type']
        model_config = self.config['model']

        logging.info(f"Loading model type: {model_type}")

        if model_type == "placeholder":
            model = PlaceholderModel(model_config)
            model.load_model()
            logging.info("Using PLACEHOLDER model (testing mode)")

        elif model_type == "yamnet_tflite":
            # TFLite 2-stage model for PYNQ
            model = YAMNetTFLiteModel(model_config)
            model_path = model_config.get('path')
            labels_path = model_config.get('labels')

            if model_path and Path(model_path).exists():
                success = model.load_model(model_path, labels_path)
                if success:
                    logging.info("YAMNet TFLite model loaded successfully")
                else:
                    logging.error("Failed to load YAMNet TFLite model")
                    logging.info("Falling back to placeholder model")
                    model = PlaceholderModel(model_config)
                    model.load_model()
            else:
                logging.warning(f"Model file not found: {model_path}")
                logging.info("Falling back to placeholder model")
                model = PlaceholderModel(model_config)
                model.load_model()

        elif model_type == "yamnet_keras":
            model = YAMNetKerasModel(model_config)
            model_path = model_config.get('path')
            labels_path = model_config.get('labels')

            if model_path and Path(model_path).exists():
                success = model.load_model(model_path, labels_path)
                if success:
                    logging.info("YAMNet+Keras model loaded successfully")
                else:
                    logging.error("Failed to load YAMNet+Keras model")
                    logging.info("Falling back to placeholder model")
                    model = PlaceholderModel(model_config)
                    model.load_model()
            else:
                logging.warning(f"Model file not found: {model_path}")
                logging.info("Falling back to placeholder model")
                model = PlaceholderModel(model_config)
                model.load_model()
        else:
            logging.error(f"Unknown model type: {model_type}")
            logging.info("Using placeholder model")
            model = PlaceholderModel(model_config)
            model.load_model()

        return model

    def load_species_targets(self) -> Dict:
        """Load target species configuration"""
        targets_path = self.config['species']['targets']

        try:
            with open(targets_path, 'r') as f:
                targets = json.load(f)
            logging.info(f"Loaded {len(targets.get('species', []))} target species")
            return targets
        except FileNotFoundError:
            logging.warning(f"Species targets file not found: {targets_path}")
            return {'species': []}
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in species targets: {e}")
            return {'species': []}

    def load_audio_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load WAV file

        Args:
            file_path: Path to WAV file

        Returns:
            Audio samples as numpy array or None if error
        """
        try:
            # Try using scipy first
            try:
                from scipy.io import wavfile
                sample_rate, audio = wavfile.read(file_path)

                # Convert to float
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0

                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                logging.info(f"Loaded audio: {len(audio)} samples at {sample_rate}Hz")
                return audio

            except ImportError:
                # Fallback to reading raw bytes
                import wave
                with wave.open(file_path, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    n_frames = wf.getnframes()
                    audio_bytes = wf.readframes(n_frames)

                    # Convert bytes to numpy array
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0

                    logging.info(f"Loaded audio: {len(audio)} samples at {sample_rate}Hz")
                    return audio

        except Exception as e:
            logging.error(f"Failed to load audio file {file_path}: {e}")
            return None

    def process_audio_file(self, audio_info: Dict):
        """
        Process a single audio file

        Args:
            audio_info: Dictionary with file information
        """
        file_path = audio_info['file_path']
        file_name = audio_info['file_name']

        logging.info("=" * 60)
        logging.info(f"Processing: {file_name}")

        try:
            # Load audio
            audio = self.load_audio_file(file_path)
            if audio is None:
                logging.error("Failed to load audio, skipping")
                return

            # Preprocess
            audio = self.model.preprocess_audio(audio, self.config['model']['sample_rate'])

            # Run inference
            logging.info("Running ML inference...")
            detections = self.model.predict(audio)

            # Filter by confidence threshold
            threshold = self.config['species']['confidence_threshold']
            filtered_detections = [d for d in detections if d['confidence'] >= threshold]

            logging.info(f"Found {len(filtered_detections)} detections above {threshold} confidence")

            # Extract and save clips if detections found
            if filtered_detections:
                for detection in filtered_detections:
                    self.save_detection_clip(audio, detection, file_name)

                    # Update stats
                    species_name = detection['class_name']
                    self.stats['species_counts'][species_name] = \
                        self.stats['species_counts'].get(species_name, 0) + 1

                self.stats['total_detections'] += len(filtered_detections)

            # Update processed count
            self.stats['files_processed'] += 1

            # Delete original file
            if self.config['storage']['delete_originals_after_processing']:
                self.storage_manager.cleanup_original(file_path, len(filtered_detections) > 0)

            # Check storage
            self.storage_manager.ensure_free_space(self.config['paths']['clips'])

            logging.info(f"Processing complete: {len(filtered_detections)} detections")

        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")

    def save_detection_clip(self, audio: np.ndarray, detection: Dict, source_file: str):
        """
        Save detection clip to file

        Args:
            audio: Full audio array
            detection: Detection dictionary
            source_file: Original filename
        """
        try:
            clip_duration = self.config['species']['clip_duration_seconds']
            sample_rate = self.config['model']['sample_rate']

            timestamp = detection['timestamp']
            start_time = max(0, timestamp - clip_duration / 2)
            end_time = min(len(audio) / sample_rate, timestamp + clip_duration / 2)

            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            clip = audio[start_sample:end_sample]

            # Generate filename
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            species = detection['class_name'].replace(' ', '_')
            confidence = detection['confidence']
            clip_filename = f"{now}_{species}_{confidence:.2f}.wav"

            clips_dir = Path(self.config['paths']['clips'])
            clip_path = clips_dir / clip_filename

            # Save clip
            from scipy.io import wavfile
            clip_int16 = (clip * 32767).astype(np.int16)
            wavfile.write(str(clip_path), sample_rate, clip_int16)

            logging.info(f"[SAVED] {clip_filename}")

        except Exception as e:
            logging.error(f"Failed to save clip: {e}")

    def start(self):
        """Start the detection system"""
        if self.running:
            logging.warning("Already running")
            return

        self.running = True

        # Scan for existing files if configured
        if self.config['processing']['scan_existing_on_startup']:
            self.audio_interface.scan_existing_files()

        # Start audio monitoring
        self.audio_interface.start()

        logging.info("=" * 60)
        logging.info("Wildlife Detection System RUNNING")
        logging.info("Waiting for audio recordings...")
        logging.info("Press Ctrl+C to stop")
        logging.info("=" * 60)

        # Main processing loop
        try:
            while self.running:
                # Get next audio file
                audio_info = self.audio_interface.get_audio(timeout=1.0)

                if audio_info:
                    self.process_audio_file(audio_info)

                # Periodic storage check
                if self.stats['files_processed'] % 10 == 0 and self.stats['files_processed'] > 0:
                    self.storage_manager.print_storage_status(
                        self.config['paths']['recordings'],
                        self.config['paths']['clips']
                    )

        except KeyboardInterrupt:
            logging.info("\nShutdown signal received")

        finally:
            self.stop()

    def stop(self):
        """Stop the detection system"""
        if not self.running:
            return

        logging.info("Stopping detection system...")
        self.running = False

        # Stop audio monitoring
        self.audio_interface.stop()

        # Print final stats
        logging.info("=" * 60)
        logging.info("FINAL STATISTICS")
        logging.info("=" * 60)
        logging.info(f"Files processed: {self.stats['files_processed']}")
        logging.info(f"Total detections: {self.stats['total_detections']}")
        logging.info(f"Species detected: {len(self.stats['species_counts'])}")
        for species, count in self.stats['species_counts'].items():
            logging.info(f"  {species}: {count}")
        logging.info("=" * 60)
        logging.info("Wildlife Detection System STOPPED")
        logging.info("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Wildlife Detection System')
    parser.add_argument('--config', default='detector_config.json',
                       help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: check config and exit')

    args = parser.parse_args()

    # Create detector
    detector = WildlifeDetector(config_path=args.config)

    if args.test:
        print("\nTest mode: Configuration loaded successfully")
        print(f"Model type: {detector.config['model']['type']}")
        print(f"Model loaded: {detector.model.is_loaded}")
        print("Exiting test mode")
        return

    # Start detection
    detector.start()


if __name__ == "__main__":
    main()
