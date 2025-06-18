from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, Union, Any

# Standard library imports
import os
import sys
import logging
from pathlib import Path
from contextlib import contextmanager
import signal
import threading
import json
import uuid

# Third-party imports
from dotenv import load_dotenv

# Add the backend directory to the path if needed
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
    # Also add the parent directory to handle both ways of running
    sys.path.append(str(current_dir.parent))

# Suppress torchaudio deprecation warnings
import warnings
warnings.filterwarnings("ignore", message="torchaudio._backend.*has been deprecated")
warnings.filterwarnings("ignore", message="torchaudio.backend.common.AudioMetaData.*has been moved")

# Initialize environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up NLTK data path
nltk_data_path = current_dir / "nltk_data"
if nltk_data_path.exists():
    import nltk
    nltk.data.path.append(str(nltk_data_path))
    logger.info(f"Added NLTK data path: {nltk_data_path}")

# Import for NLTK initialization
try:
    from .init_nltk import download_nltk_data
except ImportError:
    # Try absolute import for direct uvicorn execution
    from init_nltk import download_nltk_data

download_nltk_data()

# Import all processors
try:
    from .transcription import WhisperTranscriber
    from .summarization import ExtractiveSummarizer, AbstractiveSummarizer
    from .topic_detection import TopicDetector
    from .audio_processing import AudioProcessor
    from .youtube_downloader import YouTubeDownloader
except ImportError:
    # Try absolute imports for direct uvicorn execution
    from transcription import WhisperTranscriber
    from summarization import ExtractiveSummarizer, AbstractiveSummarizer
    from topic_detection import TopicDetector
    from audio_processing import AudioProcessor
    from youtube_downloader import YouTubeDownloader

logger.info("Imported processors successfully")

# Create uploads directory
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)

# Check for Hugging Face access token
HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
if not HF_ACCESS_TOKEN or HF_ACCESS_TOKEN == "your_huggingface_token_here":
    logger.warning("HF_ACCESS_TOKEN not set in environment variables. Speaker diarization will not work correctly.")
    logger.warning("Please set the HF_ACCESS_TOKEN environment variable with your Hugging Face access token.")
    logger.warning("You can generate a token at https://huggingface.co/settings/tokens")

# Initialize FastAPI
app = FastAPI(
    title="Podcast Processing API",
    description="API for podcast transcription, summarization, and analysis",
    version="1.0.0"
)

# Log the port the app will be running on
port = int(os.environ.get("PORT", 8000))
logger.info(f"FastAPI will run on port: {port}")

# CORS configuration
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,https://podcast-summarization-system.vercel.app").split(",")
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
ALLOWED_HEADERS = [
    "Content-Type",
    "Authorization",
    "Access-Control-Allow-Origin",
    "Access-Control-Allow-Methods",
    "Access-Control-Allow-Headers"
]

# Log warning if wildcard is used in production
if "*" in ALLOWED_ORIGINS and os.environ.get("ENVIRONMENT") == "production":
    logger.warning("WARNING: CORS is configured to allow requests from any origin (*) in production")
    logger.warning("This is not recommended for production environments")
    logger.warning("Set ALLOWED_ORIGINS env var to a comma-separated list of allowed origins")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
    expose_headers=["*"],
    max_age=3600,
)

# Define response models
class ProcessingResponse(BaseModel):
    """Response model for processing results."""
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ProcessResponse(BaseModel):
    """Response model for processing requests."""
    job_id: str
    status: str
    file_name: str
    error: Optional[str] = None

# Simplified timeout handler
class TimeoutError(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """
    Context manager for setting a timeout on operations.
    Uses threading with a timer for cross-platform compatibility.
    """
    if seconds <= 0:
        yield
        return
        
    timer = None
    timeout_occurred = False
    
    def timeout_handler():
        nonlocal timeout_occurred
        timeout_occurred = True
        logger.error(f"Operation timed out after {seconds} seconds")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.daemon = True
    timer.start()
    
    try:
        yield
        if timeout_occurred:
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
    finally:
        if timer:
            timer.cancel()

# Get timeouts from environment variables
TOPIC_DETECTION_TIMEOUT = int(os.environ.get("TOPIC_DETECTION_TIMEOUT", 120))  # 2 minutes default
SUMMARIZATION_TIMEOUT = int(os.environ.get("SUMMARIZATION_TIMEOUT", 60))  # 1 minute default

# Helper function to make objects JSON serializable
def make_serializable(obj):
    """Convert non-serializable objects to serializable types."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Health check endpoint
@app.get("/api/healthcheck")
def healthcheck():
    """
    Health check endpoint to verify API is running.
    """
    # Log the port being used
    port = os.environ.get("PORT", 8000)
    logger.info(f"API running on port: {port}")
    return {"status": "ok", "service": "Podcast Processing API", "port": port}

# Root path redirect
@app.get("/")
def read_root():
    """Redirect root path to API docs."""
    return RedirectResponse(url="/docs")

@app.get("/api")
def read_api_root():
    """Get API info."""
    return {
        "name": "Podcast Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# File cleanup helper
def cleanup_file(file_path: Path):
    """Helper function to safely cleanup uploaded files."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

# File Upload Endpoint
@app.post("/api/upload", response_model=ProcessResponse)
async def process_upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload and process an audio file
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    job_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{job_id}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        # Read and validate file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved as {file_path}")
        
        # Start processing in background
        if background_tasks:
            background_tasks.add_task(process_audio_file, file_path=file_path, job_id=job_id)
        
        return ProcessResponse(
            job_id=job_id,
            status="processing",
            file_name=file.filename
        )
    
    except Exception as e:
        # Clean up file in case of error
        cleanup_file(file_path)
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# YouTube URL Processing Endpoint
@app.post("/api/youtube", response_model=ProcessResponse)
async def process_youtube(url: str = Form(...), background_tasks: BackgroundTasks = None):
    """
    Process a YouTube video URL
    
    Args:
        url: YouTube video URL
        background_tasks: FastAPI background tasks
        
    Returns:
        ProcessResponse with job ID and status
    """
    try:
        # Create unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize YouTube downloader with longer timeout and more retries for large files
        downloader = YouTubeDownloader(
            download_dir=UPLOAD_DIR,
            max_retries=5,  # Increase retries for reliability
            timeout=60,     # Longer timeout for larger files
            chunk_size=16384  # Larger chunk size for faster downloads
        )
        
        # Validate URL first
        is_valid, error = downloader.validate_url(url)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        
        # Track download progress
        progress = {"status": "starting", "progress": 0}
        
        def progress_hook(d):
            try:
                if d['status'] == 'downloading':
                    if 'total_bytes' in d and 'downloaded_bytes' in d:
                        progress['progress'] = (d['downloaded_bytes'] / d['total_bytes']) * 100
                    progress['status'] = 'downloading'
                    progress['speed'] = d.get('speed', 0)
                    progress['eta'] = d.get('eta', 0)
                elif d['status'] == 'finished':
                    progress['status'] = 'processing'
                    progress['progress'] = 100
                elif d['status'] == 'error':
                    progress['status'] = 'error'
                    progress['error'] = d.get('error', 'Unknown error occurred')
            except Exception as e:
                logger.warning(f"Progress hook error: {str(e)}")
        
        # Add progress hook to downloader
        downloader.add_progress_hook(progress_hook)
        
        # Download audio and get metadata
        try:
            result = downloader.download_audio(url)
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                raise HTTPException(
                    status_code=408,
                    detail="Download timed out. The video might be too large or your connection might be slow. Please try again."
                )
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail="Network error occurred. Please check your internet connection and try again."
                )
            else:
                raise HTTPException(status_code=400, detail=error_msg)
        
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
            
        # Store metadata for later use
        metadata = {
            "source": "youtube",
            "title": result.get("title", ""),
            "uploader": result.get("uploader", ""),
            "duration": result.get("duration", 0),
            "duration_str": result.get("duration_str", "00:00:00"),
            "thumbnail": result.get("thumbnail", ""),
            "channel_url": result.get("channel_url", ""),
            "view_count": result.get("view_count", 0),
            "like_count": result.get("like_count", 0),
            "upload_date": result.get("upload_date", ""),
            "quality": result.get("quality", ""),
            "filesize_approx": result.get("filesize_approx", 0),
            "age_restricted": result.get("age_restricted", False),
            "availability": result.get("availability", "public"),
            "description": result.get("description", ""),
            "tags": result.get("tags", []),
            "categories": result.get("categories", []),
            "original_url": url,
            "download_progress": progress
        }
        
        # Get the file path from the result
        file_path = Path(result["file_path"])
        
        # Schedule background processing
        if background_tasks:
            background_tasks.add_task(process_audio_file, file_path, job_id, metadata)
            background_tasks.add_task(cleanup_file, file_path)
        
        return ProcessResponse(
            job_id=job_id,
            status="processing",
            file_name=file_path.name,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing YouTube URL: {str(e)}")
        error_msg = str(e)
        status_code = 400
        
        if "timeout" in error_msg.lower():
            status_code = 408
            error_msg = "Download timed out. Please try again."
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            status_code = 503
            error_msg = "Network error occurred. Please check your internet connection."
        elif "not found" in error_msg.lower():
            status_code = 404
            error_msg = "Video not found. Please check if the URL is correct."
        elif "private" in error_msg.lower():
            status_code = 403
            error_msg = "This video is private or requires authentication."
        elif "unavailable" in error_msg.lower():
            status_code = 410
            error_msg = "This video is no longer available."
        
        raise HTTPException(status_code=status_code, detail=error_msg)

# Get Results Endpoint
@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """
    Get processing results for a specific job
    """
    # Check if the results file exists
    results_file = UPLOAD_DIR / f"{job_id}_results.json"
    
    if not results_file.exists():
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "processing",
                "message": "Still processing. Please try again later."
            }
        )
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return results
    
    except Exception as e:
        logger.error(f"Error retrieving results for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

# Audio Processing Function (runs in background)
async def process_audio_file(file_path, job_id, metadata=None):
    """Process an audio file with all available processors"""
    try:
        logger.info(f"Starting processing for job {job_id} - file: {file_path}")
        
        # Initialize results dictionary
        results = {
            "job_id": job_id,
            "status": "completed",
            "error": None
        }
        
        # Create audio processor
        try:
            audio_processor = AudioProcessor()
            logger.info(f"Audio processor initialized for job {job_id}")
        except Exception as e:
            logger.error(f"Error initializing audio processor for job {job_id}: {str(e)}")
            results["warning"] = f"Error initializing audio processor: {str(e)}"
            # Continue with other processors if possible
        
        # Extract audio metadata
        try:
            audio_metadata = audio_processor.extract_metadata(file_path)
            
            if metadata:
                # Merge with provided metadata
                audio_metadata.update(metadata)
            
            # Add file info
            audio_metadata["file_path"] = str(file_path)
            audio_metadata["audio_url"] = f"/api/audio/{os.path.basename(file_path)}"
            
            # Save metadata
            results["metadata"] = audio_metadata
        except Exception as e:
            logger.error(f"Error extracting metadata for job {job_id}: {str(e)}")
            results["metadata"] = {
                "error": f"Failed to extract metadata: {str(e)}",
                "file_path": str(file_path),
                "audio_url": f"/api/audio/{os.path.basename(file_path)}"
            }
        
        # Initialize transcriber with proper error handling
        transcriber = None
        try:
            whisper_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
            transcriber = WhisperTranscriber(model_size=whisper_size)
            
            # Check if we're in mock mode (Whisper not available)
            if transcriber._mock_mode:
                logger.warning(f"Running in mock mode for job {job_id} - Whisper not available")
                results["warning"] = "Running in mock mode - Whisper not available. Results may be limited."
        except Exception as e:
            logger.error(f"Error creating transcriber for job {job_id}: {str(e)}")
            results["warning"] = f"Error initializing transcriber: {str(e)}"
            results["transcript"] = "Error: Transcription failed due to initialization error."
            results["segments"] = []
            
            # Save results early since we can't proceed with transcription
            results_file = UPLOAD_DIR / f"{job_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, default=make_serializable)
            
            logger.error(f"Processing aborted for job {job_id} - transcriber initialization failed")
            return
        
        # Transcribe audio
        try:
            if transcriber:
                logger.info(f"Transcribing audio for job {job_id}")
                transcript_result = transcriber.transcribe(file_path)
                
                # Ensure transcript_result has expected format
                if isinstance(transcript_result, dict):
                    # Save transcript
                    results["transcript"] = transcript_result.get("text", "")
                    results["segments"] = transcript_result.get("segments", [])
                else:
                    # Handle case where transcribe returns a string (shouldn't happen now but be defensive)
                    logger.warning(f"Unexpected transcript result type for job {job_id}: {type(transcript_result)}")
                    if isinstance(transcript_result, str):
                        results["transcript"] = transcript_result
                    else:
                        results["transcript"] = f"Error: Unexpected transcription result type: {type(transcript_result)}"
                    results["segments"] = []
            else:
                logger.error(f"Transcriber not available for job {job_id}")
                results["transcript"] = "Error: Transcription service unavailable."
                results["segments"] = []
        except Exception as e:
            logger.error(f"Transcription error for job {job_id}: {str(e)}")
            results["transcript"] = f"Error: {str(e)}"
            results["segments"] = []
        
        # Check if we have valid transcript before proceeding with other steps
        if not results.get("transcript") or results.get("transcript").startswith("Error:"):
            logger.warning(f"No valid transcript for job {job_id}, skipping additional processing")
            
            # Save early results and return
            results_file = UPLOAD_DIR / f"{job_id}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, default=make_serializable)
            
            logger.info(f"Processing completed with errors for job {job_id}")
            return
        
        # Topic detection (with timeout)
        if results.get("transcript") and not results.get("transcript").startswith("Error:"):
            try:
                with time_limit(TOPIC_DETECTION_TIMEOUT):
                    logger.info(f"Detecting topics for job {job_id}")
                    
                    # Process even short transcripts - adjust the minimum word count
                    if len(results["transcript"].split()) < 10:
                        logger.warning(f"Transcript very short for topic detection for job {job_id}")
                        # Create default topics for very short content
                        results["topics"] = [
                            {
                                "name": "Short Audio",
                                "keywords": ["brief", "short", "audio"],
                                "score": 1.0,
                                "color": "hsl(210, 70%, 50%)"
                            }
                        ]
                    else:
                        topic_detector = TopicDetector()
                        topics_result = topic_detector.detect_topics(results["transcript"])
                        
                        # Ensure topics have proper names
                        if isinstance(topics_result, list):
                            for i, topic in enumerate(topics_result):
                                if not topic.get("name") or topic.get("name") == "Unnamed Topic":
                                    # Create a better name based on keywords if available
                                    if topic.get("keywords") and len(topic["keywords"]) > 0:
                                        topic["name"] = f"Topic: {topic['keywords'][0].title()}"
                                    else:
                                        topic["name"] = f"Topic {i+1}"
                        
                            results["topics"] = topics_result
                        else:
                            logger.warning(f"Unexpected topic detection result type for job {job_id}: {type(topics_result)}")
                            results["topics"] = []
            except TimeoutError:
                logger.warning(f"Topic detection timed out for job {job_id}")
                results["topics"] = []
            except Exception as e:
                logger.error(f"Topic detection error for job {job_id}: {str(e)}")
                results["topics"] = []
        else:
            logger.warning(f"Skipping topic detection for job {job_id} - No valid transcript")
            results["topics"] = []
        
        # Summarization (with timeout)
        if results.get("transcript") and not results.get("transcript").startswith("Error:"):
            try:
                with time_limit(SUMMARIZATION_TIMEOUT):
                    logger.info(f"Generating summaries for job {job_id}")
                    
                    # Process even short transcripts - adjust the minimum word count
                    if len(results["transcript"].split()) < 10:
                        logger.warning(f"Transcript very short for summarization for job {job_id}")
                        results["abstractive_summary"] = "The content is too short for a meaningful summary."
                        results["extractive_summary"] = ["The content is too short for extracting key points."]
                    else:
                        # Extractive summary
                        try:
                            ext_summarizer = ExtractiveSummarizer()
                            extractive_result = ext_summarizer.summarize(results["transcript"])
                            
                            # ExtractiveSummarizer now returns a dict with 'summary' key containing a list
                            if isinstance(extractive_result, dict) and "summary" in extractive_result:
                                results["extractive_summary"] = extractive_result["summary"]
                            else:
                                logger.warning(f"Unexpected extractive summary result for job {job_id}: {type(extractive_result)}")
                                results["extractive_summary"] = ["Error generating extractive summary"]
                        except Exception as e:
                            logger.error(f"Extractive summarization error for job {job_id}: {str(e)}")
                            results["extractive_summary"] = [f"Error: {str(e)}"]
                        
                        # Abstractive summary
                        try:
                            abs_summarizer = AbstractiveSummarizer()
                            abstractive_result = abs_summarizer.summarize(results["transcript"])
                            
                            # AbstractiveSummarizer now returns a dict with 'summary' key containing a string
                            if isinstance(abstractive_result, dict) and "summary" in abstractive_result:
                                results["abstractive_summary"] = abstractive_result["summary"]
                            else:
                                logger.warning(f"Unexpected abstractive summary result for job {job_id}: {type(abstractive_result)}")
                                results["abstractive_summary"] = "Error generating abstractive summary"
                        except Exception as e:
                            logger.error(f"Abstractive summarization error for job {job_id}: {str(e)}")
                            results["abstractive_summary"] = f"Error: {str(e)}"
            except TimeoutError:
                logger.warning(f"Summarization timed out for job {job_id}")
                results["abstractive_summary"] = "Summarization timed out"
                results["extractive_summary"] = ["Summarization timed out"]
            except Exception as e:
                logger.error(f"Summarization error for job {job_id}: {str(e)}")
                results["abstractive_summary"] = f"Error: {str(e)}"
                results["extractive_summary"] = [f"Error: {str(e)}"]
        else:
            logger.warning(f"Skipping summarization for job {job_id} - No valid transcript")
            results["abstractive_summary"] = "No valid transcript for summarization"
            results["extractive_summary"] = ["No valid transcript for extractive summarization"]
        
        # Save results to file
        results_file = UPLOAD_DIR / f"{job_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, default=make_serializable)
        
        logger.info(f"Processing completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Processing error for job {job_id}: {str(e)}")
        # Save error to results file
        results_file = UPLOAD_DIR / f"{job_id}_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    "job_id": job_id,
                    "status": "error",
                    "error": str(e)
                }, f)
        except Exception as write_error:
            logger.critical(f"Failed to write error results for job {job_id}: {str(write_error)}")

# Serve audio files
@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve an audio file from the uploads directory
    """
    file_path = UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

# Add this at the end of the file
if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Render sets PORT to 10000 by default
    port = int(os.environ.get("PORT", 10000))
    host = "0.0.0.0"  # Must bind to 0.0.0.0 for Render
    
    # Print debug information
    print(f"Python version: {sys.version}")
    print(f"Starting server on http://{host}:{port}")
    print(f"Environment variables: PORT={os.environ.get('PORT')}")
    logger.info(f"BINDING TO: http://{host}:{port}")
    
    # Run the app with the correct host and port
    uvicorn.run("app:app", host=host, port=port)


