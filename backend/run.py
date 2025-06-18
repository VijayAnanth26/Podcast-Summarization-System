import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables or use defaults
host = os.environ.get("HOST", "0.0.0.0")
port = int(os.environ.get("PORT", 8000))
reload_enabled = os.environ.get("ENVIRONMENT", "development").lower() == "development"

if __name__ == "__main__":
    print(f"Starting server on http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled) 