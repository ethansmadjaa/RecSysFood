from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import auth, preferences, recommendations
import uvicorn
import asyncio
import httpx
from contextlib import asynccontextmanager
from pathlib import Path

# Keep-alive configuration
KEEP_ALIVE_INTERVAL = 600  # 10 minutes in seconds
KEEP_ALIVE_URL = "http://localhost:8000/keep-alive"

async def send_keep_alive():
    """Send periodic keep-alive requests"""
    while True:
        try:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
            async with httpx.AsyncClient() as client:
                response = await client.post(KEEP_ALIVE_URL)
                print(f"Keep-alive sent: {response.status_code}")
        except Exception as e:
            print(f"Keep-alive error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create keep-alive task
    task = asyncio.create_task(send_keep_alive())
    yield
    # Shutdown: cancel keep-alive task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="RecSysFood API", version="1.0.0", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(preferences.router)
app.include_router(recommendations.router)

# API routes
@app.get("/api")
def read_root():
    return {"message": "RecSysFood API is running", "version": "1.0.0"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.post("/keep-alive")
async def keep_alive():
    """Endpoint to receive keep-alive requests"""
    return {"status": "alive", "message": "Keep-alive received"}

@app.get("/api/test")
async def test():
    return {"status": "ok", "message": "Test request received"}

# Mount static files and serve frontend
frontend_dist = Path(__file__).parent / "frontend" / "dist"

# Serve static assets (CSS, JS, images, etc.)
app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

# Catch-all route for SPA - must be last
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve the React app for all non-API routes"""
    # If the path exists as a file, serve it
    file_path = frontend_dist / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    # Otherwise, serve index.html (for SPA routing)
    return FileResponse(frontend_dist / "index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
