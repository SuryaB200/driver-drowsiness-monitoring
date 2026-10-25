"""Single entrypoint runner for local, Codespaces, and cloud deployments."""
import os
import sys
import subprocess
from pathlib import Path

# Auto-install dependencies if uvicorn or core packages are missing
try:
    import uvicorn
    import fastapi
except ImportError:
    print("[FleetGuard] Dependencies missing. Auto-installing required packages...")
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        req_file = Path(__file__).parent / "backend" / "requirements.txt"
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print("[FleetGuard] Dependencies installed successfully!")
    except Exception as e:
        print(f"[FleetGuard] Automatic pip install failed: {e}")
        print("Please run manually: pip install -r requirements.txt")
        sys.exit(1)
    import uvicorn
    import fastapi

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("=" * 64)
    print(" FleetGuard Driver Drowsiness Monitoring System")
    print(f" Dashboard & API running at: http://{host}:{port}")
    print("=" * 64)
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
