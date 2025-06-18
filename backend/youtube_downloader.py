import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Union, Optional, Tuple
import json
import re
import yt_dlp
import hashlib
import urllib.parse
import time
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeDownloader:
    """
    YouTube downloader specifically for fetching podcast audio from YouTube.
    Focuses on extracting high-quality audio while ignoring video content.
    """
    
    def __init__(self, download_dir: Optional[Union[str, Path]] = None,
                 max_retries: int = 3,
                 timeout: int = 30,
                 chunk_size: int = 8192):
        """
        Initialize the YouTube podcast downloader.
        
        Args:
            download_dir: Optional directory to save downloads to.
            max_retries: Maximum number of retry attempts for failed downloads.
            timeout: Connection timeout in seconds.
            chunk_size: Download chunk size in bytes.
        """
        self.metadata = {}
        self.temp_files = []  # Keep track of temporary files for cleanup
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        
        # Store download directory if provided
        if download_dir:
            self.download_dir = Path(download_dir)
            os.makedirs(self.download_dir, exist_ok=True)
        else:
            self.download_dir = None
            
        # Download progress callback
        self.progress_hooks = []
        
    def add_progress_hook(self, hook):
        """Add a progress hook for download progress tracking."""
        self.progress_hooks.append(hook)
        
    def _progress_hook(self, d):
        """Internal progress hook that calls all registered hooks."""
        for hook in self.progress_hooks:
            try:
                hook(d)
            except Exception as e:
                logger.warning(f"Progress hook error: {str(e)}")
    
    def __del__(self):
        """
        Clean up any temporary files when object is destroyed.
        """
        self.cleanup()
    
    def cleanup(self):
        """
        Clean up any temporary files created during download.
        """
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    logger.info(f"Removing temporary file: {temp_file}")
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
        self.temp_files = []
    
    def validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the provided URL is a valid YouTube URL.
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not url or not isinstance(url, str):
            return False, "URL is required and must be a string"
            
        # Basic URL validation
        try:
            parsed_url = urllib.parse.urlparse(url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                return False, "Invalid URL format"
                
            # Check if it's a YouTube domain
            if not any(domain in parsed_url.netloc for domain in ["youtube.com", "youtu.be", "youtube", "ytimg.com"]):
                return False, "URL is not from a recognized YouTube domain"
                
            # Try to extract video ID as final validation
            video_id = self._extract_video_id(url)
            if not video_id or len(video_id) != 11:
                return False, "Could not extract valid YouTube video ID from URL"
                
            return True, None
        except Exception as e:
            return False, f"URL validation error: {str(e)}"
    
    def download(self, url: str, download_dir: Path) -> Path:
        """
        Download audio from a YouTube podcast episode.
        
        Args:
            url: YouTube URL of the podcast episode
            download_dir: Directory to save the downloaded audio
            
        Returns:
            Path to the downloaded audio file.
        """
        # Validate URL
        is_valid, error_message = self.validate_url(url)
        if not is_valid:
            raise ValueError(f"Invalid YouTube URL: {error_message}")
            
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Extract video ID - support both regular YouTube and Shorts URLs
        video_id = self._extract_video_id(url)
            
        output_template = str(download_dir / f"podcast_{video_id}.%(ext)s")
        
        # Configure yt-dlp options for optimal audio quality
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [self._progress_hook],
            'socket_timeout': self.timeout,
            'retries': self.max_retries,
            'fragment_retries': self.max_retries,
            'retry_sleep': lambda attempt: 2.0 ** attempt,  # Exponential backoff
            'file_access_retries': 3,
            'extractor_retries': 3,
            'http_chunk_size': self.chunk_size,
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Download the audio
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info(f"Downloading podcast audio from YouTube: {url} (Attempt {retry_count + 1}/{self.max_retries + 1})")
                    info = ydl.extract_info(url, download=True)
                    self.metadata = info
                
                # Find the downloaded file
                expected_output = download_dir / f"podcast_{video_id}.mp3"
                if expected_output.exists():
                    return expected_output
                    
                # If the expected output doesn't exist, look for any file that matches the pattern
                for file in download_dir.glob(f"podcast_{video_id}.*"):
                    return file
                    
                raise FileNotFoundError(f"Downloaded audio file not found for podcast ID: {video_id}")
                
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                last_error = e
                
                # Check for specific error types
                if "This video is unavailable" in error_msg:
                    logger.error(f"The YouTube video is unavailable or private: {error_msg}")
                    raise ValueError(f"The YouTube video is unavailable or private. Please check if the video exists and is public.")
                elif "Video unavailable" in error_msg:
                    logger.error(f"The YouTube video is unavailable: {error_msg}")
                    raise ValueError(f"The YouTube video is unavailable. It may have been removed or is region-restricted.")
                elif "Sign in" in error_msg:
                    logger.error(f"The YouTube video requires authentication: {error_msg}")
                    raise ValueError(f"The YouTube video requires authentication. Age-restricted or private videos cannot be downloaded.")
                elif any(x in error_msg.lower() for x in ["timeout", "timed out", "connection", "network"]):
                    # Network-related errors - retry
                    retry_count += 1
                    if retry_count <= self.max_retries:
                        wait_time = 2.0 ** retry_count  # Exponential backoff
                        logger.warning(f"Network error, retrying in {wait_time:.1f} seconds... ({retry_count}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for network errors")
                        raise ValueError(f"Failed to download after {self.max_retries} attempts: Network issues")
                else:
                    logger.error(f"Error downloading podcast audio: {error_msg}")
                    raise ValueError(f"Error downloading from YouTube: {error_msg}")
                    
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count <= self.max_retries:
                    wait_time = 2.0 ** retry_count
                    logger.warning(f"Download error, retrying in {wait_time:.1f} seconds... ({retry_count}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries ({self.max_retries}) exceeded")
                    raise
        
        # If we get here, all retries failed
        raise ValueError(f"Failed to download after {self.max_retries} attempts: {str(last_error)}")
    
    def _extract_video_id(self, url: str) -> str:
        """
        Extract YouTube video ID from URL supporting various formats.
        
        Args:
            url: YouTube URL
            
        Returns:
            YouTube video ID
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/e\/|youtube\.com\/user\/.*\/.*\/|youtube\.com\/user\/.*\?v=|youtube\.com\/shorts\/|youtube\.com\/live\/|youtube\.com\/\?v=|youtube\.com\/watch\?.*v=)([^&\n?#]+)',
            r'(?:youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If URL doesn't match pattern, assume it's a video ID or generate hash
        if re.match(r'^[A-Za-z0-9_-]{11}$', url):
            return url
        else:
            # Use a hash of the URL as fallback
            return hashlib.md5(url.encode()).hexdigest()[:11]
    
    def get_metadata(self) -> dict:
        """
        Get detailed metadata from the downloaded podcast.
        
        Returns:
            Dictionary with podcast metadata
        """
        if not self.metadata:
            return {
                "title": "",
                "channel": "",
                "channel_id": "",
                "channel_url": "",
                "view_count": 0,
                "like_count": 0,
                "upload_date": "",
                "duration": 0,
                "thumbnail": "",
                "description": "",
                "tags": [],
                "categories": [],
                "quality": "",
                "filesize_approx": 0,
                "is_live": False,
                "was_live": False,
                "age_restricted": False,
                "availability": "public",
                "channel_follower_count": 0,
                "comment_count": 0,
                "format": "",
                "original_url": "",
                "webpage_url": "",
            }
            
        # Format duration in HH:MM:SS
        duration_secs = self.metadata.get("duration", 0)
        hours = duration_secs // 3600
        minutes = (duration_secs % 3600) // 60
        seconds = duration_secs % 60
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Format date as YYYY-MM-DD
        upload_date = self.metadata.get("upload_date", "")
        if upload_date and len(upload_date) == 8:
            upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
            
        # Get best quality thumbnail
        thumbnails = self.metadata.get("thumbnails", [])
        best_thumbnail = ""
        max_width = 0
        if thumbnails:
            for thumb in thumbnails:
                width = thumb.get("width", 0)
                if width > max_width:
                    max_width = width
                    best_thumbnail = thumb.get("url", "")
        
        # Get audio format details
        format_info = self.metadata.get("format", "")
        if not format_info:
            formats = self.metadata.get("formats", [])
            if formats:
                # Get the best audio format
                audio_formats = [f for f in formats if f.get("acodec", "none") != "none"]
                if audio_formats:
                    best_audio = max(audio_formats, key=lambda x: x.get("abr", 0))
                    format_info = f"{best_audio.get('format_note', '')} {best_audio.get('ext', '')} @ {best_audio.get('abr', 0)}kbps"
        
        return {
            "title": self.metadata.get("title", ""),
            "channel": self.metadata.get("uploader", ""),
            "channel_id": self.metadata.get("channel_id", ""),
            "channel_url": self.metadata.get("channel_url", ""),
            "view_count": self.metadata.get("view_count", 0),
            "like_count": self.metadata.get("like_count", 0),
            "upload_date": upload_date,
            "duration": duration_secs,
            "duration_str": duration_str,
            "thumbnail": best_thumbnail or self.metadata.get("thumbnail", ""),
            "description": self.metadata.get("description", ""),
            "tags": self.metadata.get("tags", []),
            "categories": self.metadata.get("categories", []),
            "quality": format_info,
            "filesize_approx": self.metadata.get("filesize_approx", 0),
            "is_live": self.metadata.get("is_live", False),
            "was_live": self.metadata.get("was_live", False),
            "age_restricted": self.metadata.get("age_limit", 0) > 0,
            "availability": self.metadata.get("availability", "public"),
            "channel_follower_count": self.metadata.get("channel_follower_count", 0),
            "comment_count": self.metadata.get("comment_count", 0),
            "format": format_info,
            "original_url": self.metadata.get("original_url", ""),
            "webpage_url": self.metadata.get("webpage_url", ""),
        }
    
    def download_audio(self, url: str) -> Dict:
        """
        Download audio from a YouTube video and return relevant metadata.
        
        Args:
            url: YouTube URL of the video
            
        Returns:
            Dictionary containing file_path and metadata
        """
        if not self.download_dir:
            raise ValueError("No download directory specified. Either provide one during initialization or use the download method directly.")
            
        try:
            # Download the audio file
            file_path = self.download(url, self.download_dir)
            
            # Get metadata
            metadata = self.get_metadata()
            
            # Return result dictionary with enhanced metadata
            return {
                "file_path": str(file_path),
                "title": metadata.get("title", ""),
                "uploader": metadata.get("channel", ""),
                "duration": metadata.get("duration", 0),
                "duration_str": metadata.get("duration_str", "00:00:00"),
                "thumbnail": metadata.get("thumbnail", ""),
                "channel_url": metadata.get("channel_url", ""),
                "view_count": metadata.get("view_count", 0),
                "like_count": metadata.get("like_count", 0),
                "upload_date": metadata.get("upload_date", ""),
                "quality": metadata.get("quality", ""),
                "filesize_approx": metadata.get("filesize_approx", 0),
                "age_restricted": metadata.get("age_restricted", False),
                "availability": metadata.get("availability", "public"),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "categories": metadata.get("categories", []),
                "error": None
            }
        except Exception as e:
            logger.error(f"Error downloading audio from YouTube: {str(e)}")
            return {
                "file_path": None,
                "error": str(e),
                "title": "",
                "uploader": "",
                "duration": 0,
                "duration_str": "00:00:00",
                "thumbnail": "",
                "channel_url": "",
                "view_count": 0,
                "like_count": 0,
                "upload_date": "",
                "quality": "",
                "filesize_approx": 0,
                "age_restricted": False,
                "availability": "unknown",
                "description": "",
                "tags": [],
                "categories": []
            } 
