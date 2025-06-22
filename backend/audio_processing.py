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
            
    def trim_audio(self, file_path: Union[str, Path], start_time: float, end_time: float, 
                  output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Trim audio/video file based on start and end times.
        
        Args:
            file_path: Path to the input file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Optional path for output file. If not provided, a temporary file is created.
            
        Returns:
            Path to the trimmed file
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Validate start and end times
            if start_time < 0:
                start_time = 0
                
            # Get file metadata to check duration
            metadata = self.extract_metadata(file_path)
            if end_time > metadata.duration:
                end_time = metadata.duration
                
            if start_time >= end_time:
                raise ValueError(f"Invalid time range: start_time ({start_time}) must be less than end_time ({end_time})")
                
            # Create output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extension = file_path.suffix
                output_path = self.cache_dir / f"trim_{timestamp}_{start_time:.2f}_{end_time:.2f}{extension}"
            else:
                output_path = Path(output_path)
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate duration
            duration = end_time - start_time
            
            # Use ffmpeg to trim the file
            logger.info(f"Trimming {file_path} from {start_time:.2f}s to {end_time:.2f}s")
            
            # Determine if it's a video or audio file
            is_video = any(stream.get('codec_type') == 'video' 
                          for stream in ffmpeg.probe(file_path)['streams'])
            
            # Set up ffmpeg command
            input_stream = ffmpeg.input(str(file_path), ss=start_time, t=duration)
            
            if is_video:
                # For video files, maintain both audio and video
                output_stream = ffmpeg.output(
                    input_stream, 
                    str(output_path),
                    c='copy',  # Use copy codec for faster processing
                    **self.ffmpeg_options
                )
            else:
                # For audio files
                output_stream = ffmpeg.output(
                    input_stream,
                    str(output_path),
                    acodec='copy',  # Use copy codec for faster processing
                    **self.ffmpeg_options
                )
                
            # Run the ffmpeg command
            output_stream.run(quiet=True)
            
            logger.info(f"Trimmed file saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error trimming file: {str(e)}")
            raise
