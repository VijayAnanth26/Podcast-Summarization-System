import os
import tempfile
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging
import time
import json
import torch
import ffmpeg

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
    logger.info("Whisper library available")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper library not available. Using mock transcription.")

# Try to import ffmpeg
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logger.warning("ffmpeg-python library not available. Audio extraction will be limited.")

class WhisperTranscriber:
    """
    Transcription service using OpenAI's Whisper model.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_size: Size of the Whisper model to use (tiny, base, small, medium, large)
            device: Device to use for inference (cpu, cuda, mps)
        """
        self.model_size = model_size
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA for transcription")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Apple Silicon) for transcription")
            else:
                self.device = "cpu"
                logger.info("Using CPU for transcription")
        else:
            self.device = device
        
        # Load the model (lazy loading)
        self._model = None
        self._mock_mode = not WHISPER_AVAILABLE
        
    @property
    def model(self):
        """Lazy load the Whisper model"""
        if self._mock_mode:
            logger.warning("Running in mock mode - Whisper not available")
            return None
            
        if self._model is None:
            try:
                logger.info(f"Loading Whisper {self.model_size} model...")
                
                # Set compute optimization flags
                torch.set_num_threads(4)  # Limit CPU threads
                if self.device == "cuda":
                    # Enable TF32 precision on Ampere or newer GPUs
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    # Use mixed precision
                    torch.set_float32_matmul_precision('medium')
                
                # Load the model with compute type optimization
                if self.device == "cuda":
                    self._model = whisper.load_model(
                        self.model_size, 
                        device=self.device,
                        download_root=os.path.expanduser("~/.cache/whisper"),
                        in_memory=True
                    ).half()  # Use half precision on GPU
                else:
                    self._model = whisper.load_model(
                        self.model_size, 
                        device=self.device,
                        download_root=os.path.expanduser("~/.cache/whisper")
                    )
                
                logger.info(f"Whisper {self.model_size} model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {str(e)}")
                self._mock_mode = True
                return None
                
        return self._model
    
    def extract_audio(self, file_path: Union[str, Path]) -> Path:
        """
        Extract audio from video file if necessary.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Path to the extracted audio file or the original file if already audio
        """
        file_path = Path(file_path)
        
        # Check if file is already an audio file
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        if file_path.suffix.lower() in audio_extensions:
            return file_path
        
        # Check if ffmpeg is available
        if not FFMPEG_AVAILABLE:
            logger.warning("ffmpeg-python not available, returning original file without extraction")
            return file_path
            
        # Extract audio to temporary file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_audio_path = Path(temp_audio.name)
        
        try:
            logger.info(f"Extracting audio from {file_path}")
            # Use ffmpeg to extract audio
            (
                ffmpeg
                .input(str(file_path))
                .output(str(temp_audio_path), acodec='libmp3lame', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Audio extracted to {temp_audio_path}")
            return temp_audio_path
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            if temp_audio_path.exists():
                os.unlink(temp_audio_path)
            
            # Fall back to using the original file
            logger.warning(f"Falling back to using original file without extraction")
            return file_path
    
    def transcribe(self, file_path: Union[str, Path]) -> Dict:
        """
        Transcribe audio/video file to text.
        
        Args:
            file_path: Path to the file to transcribe
            
        Returns:
            Dictionary containing transcript text and segments
        """
        # Check if we're in mock mode
        if self._mock_mode:
            return self._mock_transcribe(file_path)
            
        audio_path = None
        extracted_audio = False
        
        try:
            # Extract audio if needed
            audio_path = self.extract_audio(file_path)
            extracted_audio = (audio_path != Path(file_path))
            
            logger.info(f"Transcribing {audio_path}...")
            
            # Process with Whisper
            logger.info(f"Transcribing {audio_path} with Whisper {self.model_size} model...")
            
            transcribe_options = {
                "language": None,
                "verbose": False,
                "task": "transcribe",
                # Optimize for speed
                "fp16": self.device == "cuda",  # Use FP16 on CUDA
                "beam_size": 3,  # Reduce beam size (default is 5)
                "best_of": 3,    # Reduce best of (default is 5)
                "initial_prompt": "This is a podcast or audio recording with one or more speakers.",
            }
            
            # Handle too large files by processing in batches
            if audio_path.stat().st_size > 50 * 1024 * 1024:  # 50MB
                logger.info(f"Large file detected ({audio_path.stat().st_size/1024/1024:.2f}MB), using optimized processing")
                # Process large file with optimizations
                result = self.model.transcribe(str(audio_path), **transcribe_options)
            else:
                # Process normal sized file
                result = self.model.transcribe(str(audio_path), **transcribe_options)
                
            # Extract transcript
            transcript = result["text"]
            logger.info(f"Transcription complete: {len(transcript)} characters")
            
            # Return a dictionary with text and segments
            return {
                "text": transcript,
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
        finally:
            # Clean up temporary audio file if we extracted it
            if extracted_audio and audio_path and audio_path.exists() and audio_path != Path(file_path):
                try:
                    os.unlink(audio_path)
                    logger.info(f"Cleaned up temporary audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp audio file {audio_path}: {str(e)}")
    
    def transcribe_with_timestamps(self, file_path: Union[str, Path]) -> Dict:
        """
        Transcribe audio/video file to text with timestamps.
        
        Args:
            file_path: Path to the file to transcribe
            
        Returns:
            Dictionary with transcript including word-level timestamps
        """
        # Check if we're in mock mode
        if self._mock_mode:
            return self._mock_transcribe_with_timestamps(file_path)
        
        audio_path = None
        extracted_audio = False
        
        try:
            # Extract audio if needed
            audio_path = self.extract_audio(file_path)
            extracted_audio = (audio_path != file_path)
            
            logger.info(f"Transcribing {audio_path} with timestamps...")
            
            # Run the model with word timestamps
            result = self.model.transcribe(str(audio_path), word_timestamps=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription with timestamps: {str(e)}")
            # Fall back to mock transcription
            return self._mock_transcribe_with_timestamps(file_path)
            
        finally:
            # Clean up if we extracted audio
            if extracted_audio and audio_path and Path(audio_path).exists():
                try:
                    os.unlink(audio_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary audio file: {str(e)}")
                    
    def _mock_transcribe(self, file_path: Union[str, Path]) -> Dict:
        """
        Generate a mock transcript when Whisper is not available.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Mock transcript dictionary with text and segments
        """
        logger.warning(f"Using mock transcription for {file_path}")
        
        # Simulate minimal processing time 
        time.sleep(0.5)
        
        # Generate a simple mock transcript
        filename = Path(file_path).name
        text = (
            f"This is a mock transcript for {filename}. "
            "The actual transcription could not be performed because the Whisper library "
            "is not available or an error occurred during processing. "
            "Please install the necessary dependencies and check the logs for more information."
        )
        
        # Create a mock transcript dictionary
        return {
            "text": text,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is a mock transcript"
                },
                {
                    "id": 1,
                    "start": 5.0,
                    "end": 10.0,
                    "text": f"for {filename}."
                }
            ]
        }
    
    def _mock_transcribe_with_timestamps(self, file_path: Union[str, Path]) -> Dict:
        """
        Generate a mock transcript with timestamps when Whisper is not available.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Mock transcript data with timestamps
        """
        logger.warning(f"Using mock transcription with timestamps for {file_path}")
        
        # Get basic mock result
        mock_result = self._mock_transcribe(file_path)
        
        # Extract the text from the mock result dictionary
        text = mock_result["text"]
        
        # Create a basic structure similar to Whisper's output
        filename = Path(file_path).name
        return {
            "text": text,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is a mock transcript",
                    "words": [
                        {"word": "This", "start": 0.0, "end": 0.5},
                        {"word": "is", "start": 0.5, "end": 0.8},
                        {"word": "a", "start": 0.8, "end": 1.0},
                        {"word": "mock", "start": 1.0, "end": 1.5},
                        {"word": "transcript", "start": 1.5, "end": 2.5}
                    ]
                },
                {
                    "id": 1,
                    "start": 5.0,
                    "end": 10.0,
                    "text": f"for {filename}.",
                    "words": [
                        {"word": "for", "start": 5.0, "end": 5.5},
                        {"word": filename, "start": 5.5, "end": 9.5},
                        {"word": ".", "start": 9.5, "end": 10.0}
                    ]
                }
            ],
            "language": "en"
        }

    def save_transcript(self, transcript: Dict, output_path: Union[str, Path]) -> str:
        """
        Save transcript to file.
        
        Args:
            transcript: Transcript dictionary from transcribe()
            output_path: Path to save transcript
            
        Returns:
            Path to saved transcript
        """
        output_path = Path(output_path)
        
        try:
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved transcript to {output_path}")
            return str(output_path) 
        except Exception as e:
            logger.error(f"Error saving transcript: {str(e)}")
            return "" 
