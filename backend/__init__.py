# Backend package for podcast processing pipeline
# 
# This package provides a comprehensive set of tools for processing podcast audio:
# - Audio processing and metadata extraction
# - Transcription using Whisper
# - Summarization (both extractive and abstractive)
# - Topic detection and keyword extraction
# - YouTube downloading capabilities
#
# Each module is designed to work independently or as part of the pipeline.

import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import modules to make them available when importing the package
try:
    from .transcription import WhisperTranscriber
    logger.info("Successfully imported WhisperTranscriber")
except Exception as e:
    logger.warning(f"Failed to import WhisperTranscriber: {e}")

try:
    from .summarization import ExtractiveSummarizer, AbstractiveSummarizer
    logger.info("Successfully imported Summarizers")
except Exception as e:
    logger.warning(f"Failed to import Summarizers: {e}")

try:
    from .topic_detection import TopicDetector
    logger.info("Successfully imported TopicDetector")
except Exception as e:
    logger.warning(f"Failed to import TopicDetector: {e}")

try:
    from .audio_processing import AudioProcessor
    logger.info("Successfully imported AudioProcessor")
except Exception as e:
    logger.warning(f"Failed to import AudioProcessor: {e}")

try:
    from .youtube_downloader import YouTubeDownloader
    logger.info("Successfully imported YouTubeDownloader")
except Exception as e:
    logger.warning(f"Failed to import YouTubeDownloader: {e}")

# Define what's available when doing "from backend import *"
__all__ = [
    'WhisperTranscriber',
    'ExtractiveSummarizer',
    'AbstractiveSummarizer',
    'TopicDetector',
    'AudioProcessor',
    'YouTubeDownloader'
] 
