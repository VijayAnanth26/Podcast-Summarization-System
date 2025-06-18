import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Union, Optional, List, Any
import ffmpeg
from datetime import datetime
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioMetadata(BaseModel):
    """Data model for audio metadata"""
    duration: float
    format: str
    channels: int
    sample_rate: int
    bit_rate: Optional[int] = None
    file_size: int
    codec: str
    tags: Dict[str, str] = {}

class AudioProcessor:
    """Audio processing utilities for podcast editing"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the audio processor.
        
        Args:
            cache_dir: Directory to store temporary files and cache
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "audio_processor"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure ffmpeg options
        self.ffmpeg_options = {
            'loglevel': 'error',
            'hide_banner': None,
            'y': None  # Overwrite output files
        }

    def extract_metadata(self, file_path: Union[str, Path]) -> AudioMetadata:
        """
        Extract comprehensive metadata from audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioMetadata object containing file information
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # Get detailed probe information
            probe = ffmpeg.probe(file_path)
            
            # Get audio stream info
            audio_info = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if not audio_info:
                raise ValueError("No audio stream found in file")

            # Extract format information
            fmt_info = probe['format']
            
            return AudioMetadata(
                duration=float(fmt_info['duration']),
                format=fmt_info['format_name'],
                channels=int(audio_info.get('channels', 2)),
                sample_rate=int(audio_info.get('sample_rate', 44100)),
                bit_rate=int(audio_info.get('bit_rate', 0)),
                file_size=int(fmt_info['size']),
                codec=audio_info.get('codec_name', 'unknown'),
                tags=fmt_info.get('tags', {})
            )

        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise
