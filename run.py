"""Single entrypoint runner for local and cloud deployments."""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting FleetGuard Driver Drowsiness Monitoring on http://{host}:{port}...")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
