import logging
import nltk
import ssl
import os
import shutil
from pathlib import Path
import time
import contextlib
import threading

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default NLTK data directory within the package
DEFAULT_NLTK_DATA = Path(__file__).parent / "nltk_data"
os.makedirs(DEFAULT_NLTK_DATA, exist_ok=True)

# Create a lock file for NLTK downloads to prevent concurrent downloads
LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nltk_download.lock")

@contextlib.contextmanager
def download_lock(timeout=60):
    """
    File-based lock to prevent concurrent NLTK data downloads.
    """
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            with open(LOCK_FILE, 'x') as f:
                f.write(f"PID: {os.getpid()}, Time: {time.ctime()}")
            try:
                yield
            finally:
                if os.path.exists(LOCK_FILE):
                    os.unlink(LOCK_FILE)
            return
        except FileExistsError:
            # Lock file exists, check if it's stale
            try:
                mtime = os.path.getmtime(LOCK_FILE)
                if time.time() - mtime > 300:  # 5 minutes
                    logger.warning("Removing stale NLTK download lock file")
                    os.unlink(LOCK_FILE)
                    continue
            except OSError:
                pass
            time.sleep(1)
    raise TimeoutError("Timed out waiting for NLTK download lock")

def download_nltk_data():
    """
    Download required NLTK data packages.
    Uses a timeout to prevent hanging on slow connections.
    Falls back to offline mode if network is unavailable.
    Checks for a local copy of NLTK data first.
    """
    try:
        # Set up the NLTK data directory
        nltk_data_dir = os.environ.get("NLTK_DATA")
        if not nltk_data_dir:
            nltk_data_dir = str(Path.home() / "nltk_data")
            os.makedirs(nltk_data_dir, exist_ok=True)
            os.environ["NLTK_DATA"] = nltk_data_dir

        # Add the default local directory to NLTK's search path
        nltk.data.path.insert(0, str(DEFAULT_NLTK_DATA))
        
        # Also add the package directory to NLTK's search path
        package_nltk_dir = Path(__file__).parent / "nltk_data"
        if package_nltk_dir.exists() and str(package_nltk_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(package_nltk_dir))
            logger.info(f"Added package NLTK data path: {package_nltk_dir}")
        
        # Check if we're offline by trying a basic connection
        network_available = False
        try:
            import urllib.request
            urllib.request.urlopen("https://www.google.com", timeout=2)
            network_available = True
        except:
            logger.warning("Network connectivity check failed. Working in offline mode.")
        
        with download_lock():
            logger.info("Setting up NLTK data packages...")
            
            # Required packages
            required_packages = ['punkt', 'stopwords']
            missing_required = []
            
            # Check which packages are already available locally
            for package in required_packages:
                try:
                    if package == 'punkt':
                        nltk.data.find(f"tokenizers/{package}")
                        logger.info(f"Package '{package}' is already available.")
                    elif package == 'stopwords':
                        nltk.data.find(f"corpora/{package}")
                        logger.info(f"Package '{package}' is already available.")
                except LookupError:
                    missing_required.append(package)
            
            # If we're missing packages but have no network, try to create local versions
            if missing_required and not network_available:
                # Check for bundled data files
                for package in missing_required[:]:
                    # Check prepackaged data
                    logger.info(f"Looking for bundled '{package}' data...")
                    
                    # If we have a prepackaged version, use it
                    if (Path(__file__).parent / f"data/{package}").exists():
                        try:
                            dest_dir = DEFAULT_NLTK_DATA
                            if package == 'punkt':
                                os.makedirs(dest_dir / "tokenizers", exist_ok=True)
                                shutil.copytree(
                                    Path(__file__).parent / f"data/{package}", 
                                    dest_dir / "tokenizers" / package
                                )
                            elif package == 'stopwords':
                                os.makedirs(dest_dir / "corpora", exist_ok=True) 
                                shutil.copytree(
                                    Path(__file__).parent / f"data/{package}", 
                                    dest_dir / "corpora" / package
                                )
                            logger.info(f"Installed bundled '{package}' data successfully.")
                            missing_required.remove(package)
                        except Exception as e:
                            logger.warning(f"Error installing bundled '{package}' data: {e}")
            
            # If we have network, try to download missing packages
            if missing_required and network_available:
                # Work around SSL certificate issues when downloading
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                
                # Download missing required packages
                for package in missing_required:
                    try:
                        logger.info(f"Downloading required package '{package}'...")
                        # Try multiple times with increasing timeouts
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            try:
                                timeout = 5 * (attempt + 1)  # 5, 10, 15 seconds
                                
                                # Use a thread with timeout to prevent hanging
                                download_success = [False]
                                download_error = [None]
                                
                                def download_with_timeout():
                                    try:
                                        nltk.download(package, download_dir=str(DEFAULT_NLTK_DATA), quiet=True)
                                        download_success[0] = True
                                    except Exception as e:
                                        download_error[0] = e
                                
                                download_thread = threading.Thread(target=download_with_timeout)
                                download_thread.daemon = True
                                download_thread.start()
                                download_thread.join(timeout)
                                
                                if download_success[0]:
                                    logger.info(f"Required package '{package}' downloaded successfully.")
                                    break
                                elif download_error[0]:
                                    raise download_error[0]
                                else:
                                    raise TimeoutError(f"Download timed out after {timeout} seconds")
                            except Exception as e:
                                if attempt == max_attempts - 1:
                                    raise
                                logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                                time.sleep(1)
                    except Exception as e:
                        logger.warning(f"Error downloading required package '{package}': {str(e)}")
                        logger.warning("This may affect some functionality, but the application will continue.")
            
            # Optional packages - don't fail if they can't be downloaded
            logger.info("Checking optional NLTK data packages...")
            optional_packages = ['wordnet', 'averaged_perceptron_tagger']
            
            for package in optional_packages:
                try:
                    # Try to find the package first
                    package_path = "corpora/" + package
                    if package == "averaged_perceptron_tagger":
                        package_path = "taggers/" + package
                    
                    try:
                        nltk.data.find(package_path)
                        logger.info(f"Optional package '{package}' is already available.")
                        continue
                    except LookupError:
                        pass
                        
                    # Check for bundled wordnet.zip file
                    if package == 'wordnet':
                        bundled_wordnet = Path(__file__).parent / "nltk_data/corpora/wordnet.zip"
                        if bundled_wordnet.exists():
                            try:
                                # Create the destination directory
                                wordnet_dir = Path(nltk_data_dir) / "corpora"
                                os.makedirs(wordnet_dir, exist_ok=True)
                                
                                # Copy the file to the NLTK data directory
                                shutil.copy(bundled_wordnet, wordnet_dir / "wordnet.zip")
                                logger.info(f"Installed bundled wordnet.zip successfully")
                                continue
                            except Exception as e:
                                logger.warning(f"Error installing bundled wordnet.zip: {e}")
                    
                    # If we don't have network, skip download
                    if not network_available:
                        logger.info(f"Skipping optional package '{package}' download in offline mode.")
                        continue
                    
                    # Try to download
                    logger.info(f"Downloading optional package '{package}'...")
                    
                    # Use a thread with timeout to prevent hanging
                    download_success = [False]
                    download_error = [None]
                    
                    def download_package():
                        try:
                            nltk.download(package, download_dir=str(DEFAULT_NLTK_DATA), quiet=True, raise_on_error=False)
                            download_success[0] = True
                        except Exception as e:
                            download_error[0] = e
                    
                    download_thread = threading.Thread(target=download_package)
                    download_thread.daemon = True
                    download_thread.start()
                    
                    # Wait for download with timeout
                    timeout = 20  # 20 second timeout for optional packages
                    download_thread.join(timeout)
                    
                    if download_success[0]:
                        logger.info(f"Optional package '{package}' downloaded successfully.")
                    elif download_error[0]:
                        logger.info(f"Could not download optional package '{package}': {str(download_error[0])}")
                    else:
                        logger.info(f"Download of optional package '{package}' timed out.")
                except Exception as e:
                    logger.info(f"Optional package '{package}' not available: {str(e)}")
            
            # Check if we have the minimum required packages for functionality
            missing_required = []
            for package in required_packages:
                try:
                    if package == 'punkt':
                        nltk.data.find(f"tokenizers/{package}")
                    elif package == 'stopwords':
                        nltk.data.find(f"corpora/{package}")
                except LookupError:
                    missing_required.append(package)
            
            if missing_required:
                logger.warning(f"Missing required NLTK packages: {', '.join(missing_required)}")
                logger.warning("Some functionality may be limited. Try running this script with internet connectivity.")
            else:
                logger.info("NLTK data packages setup completed successfully.")
            
            return True
    except Exception as e:
        logger.error(f"Error setting up NLTK data: {str(e)}")
        return False

if __name__ == "__main__":
    download_nltk_data() 
