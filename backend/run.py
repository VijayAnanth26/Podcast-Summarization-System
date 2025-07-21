import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables or use defaults
host = os.environ.get("HOST", "0.0.0.0")
port = int(os.environ.get("PORT", 10000))  # Default to 10000 for Render
environment = os.environ.get("ENVIRONMENT", "development").lower()
reload_enabled = environment == "development"

if __name__ == "__main__":
    # Log startup information
    print(f"Starting server on http://{host}:{port}")
    print(f"Environment: {environment}")
    
    # Start server with the correct port
    # Disable reload in production environments
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled) 