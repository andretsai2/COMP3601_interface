"""
YAMNet TFLite Model - 2-Stage Pipeline
Uses YAMNet embeddings + custom classifier for wildlife detection
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from src.base_model import BaseWildlifeModel

logger = logging.getLogger(__name__)


class YAMNetTFLiteModel(BaseWildlifeModel):
    """
    Wildlife detection using 2-stage TFLite pipeline:
    1. YAMNet embeddings (waveform → 1024-d)
    2. Custom classifier (1024-d → 12 classes)
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        self.yamnet_interpreter = None
        self.classifier_interpreter = None
        self.yamnet_input_details = None
        self.yamnet_output_details = None
        self.classifier_input_details = None
        self.classifier_output_details = None
        self.class_names = []

        logger.info("YAMNet TFLite 2-stage model initializing...")

    def load_model(self, model_path: str, labels_path: Optional[str] = None) -> bool:
        """
        Load both YAMNet embedding model and classifier

        Args:
            model_path: Path to classifier .tflite file
            labels_path: Path to class labels file
        """
        try:
            import tflite_runtime.interpreter as tflite

            # YAMNet embedding model path
            yamnet_path = "models/yamnet_embedding.tflite"

            logger.info("Loading YAMNet embedding model...")
            self.yamnet_interpreter = tflite.Interpreter(model_path=yamnet_path)
            self.yamnet_interpreter.allocate_tensors()
            self.yamnet_input_details = self.yamnet_interpreter.get_input_details()
            self.yamnet_output_details = self.yamnet_interpreter.get_output_details()
            logger.info("✓ YAMNet embeddings loaded")

            logger.info(f"Loading classifier: {model_path}")
            self.classifier_interpreter = tflite.Interpreter(model_path=model_path)
            self.classifier_interpreter.allocate_tensors()
            self.classifier_input_details = self.classifier_interpreter.get_input_details()
            self.classifier_output_details = self.classifier_interpreter.get_output_details()
            logger.info("✓ Classifier loaded")

            # Load labels
            if labels_path:
                with open(labels_path, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"✓ Loaded {len(self.class_names)} class labels")
            else:
                num_classes = self.classifier_output_details[0]['shape'][-1]
                self.class_names = [f"Class_{i}" for i in range(num_classes)]

            self.is_loaded = True
            return True

        except ImportError:
            logger.error("tflite_runtime not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio for YAMNet"""
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        # Normalize
        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            audio = audio / (audio_max + 1e-8)

        # Resample if needed
        if sample_rate != 16000:
            logger.info(f"Resampling from {sample_rate}Hz to 16kHz")
            duration = len(audio) / sample_rate
            new_len = int(duration * 16000)
            audio = np.interp(
                np.linspace(0, len(audio)-1, new_len),
                np.arange(len(audio)),
                audio
            ).astype(np.float32)

        return audio

    def extract_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """Extract YAMNet embeddings from audio"""
        # Resize input tensor
        input_index = self.yamnet_input_details[0]['index']
        self.yamnet_interpreter.resize_tensor_input(input_index, [len(audio)])
        self.yamnet_interpreter.allocate_tensors()

        # Run YAMNet
        self.yamnet_interpreter.set_tensor(input_index, audio)
        self.yamnet_interpreter.invoke()

        # Get embeddings
        output_index = self.yamnet_output_details[0]['index']
        embeddings = self.yamnet_interpreter.get_tensor(output_index)

        # Average across time frames
        if embeddings.ndim == 2:
            embedding = np.mean(embeddings, axis=0)
        else:
            embedding = embeddings.flatten()
            if len(embedding) % 1024 == 0:
                embedding = embedding.reshape(-1, 1024).mean(axis=0)

        return embedding.astype(np.float32)

    def predict(self, audio: np.ndarray) -> List[Dict]:
        """Run inference on audio"""
        if not self.is_loaded:
            logger.error("Model not loaded!")
            return []

        if not self.validate_audio(audio):
            logger.error("Invalid audio input")
            return []

        try:
            # Extract embeddings
            embedding = self.extract_embeddings(audio)

            # Reshape for classifier
            embedding_input = embedding.reshape(1, 1024)

            # Run classifier
            input_index = self.classifier_input_details[0]['index']
            output_index = self.classifier_output_details[0]['index']

            self.classifier_interpreter.set_tensor(input_index, embedding_input)
            self.classifier_interpreter.invoke()

            # Get predictions
            probs = self.classifier_interpreter.get_tensor(output_index).squeeze()
            probs = probs / (np.sum(probs) + 1e-8)

            # Get top prediction
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            detection = {
                'class_id': predicted_class,
                'class_name': self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"Class_{predicted_class}",
                'confidence': confidence,
                'timestamp': 0.0,
                'all_probabilities': probs.tolist()
            }

            logger.info(f"Prediction: {detection['class_name']} ({confidence:.3f})")

            return [detection]

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_model_info(self) -> Dict:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            'model_type': 'tflite_2stage',
            'num_classes': len(self.class_names),
            'class_names': self.class_names
        })
        return info
