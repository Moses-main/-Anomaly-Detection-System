"""
Network Anomaly Detection System - Main Application
FastAPI backend with real-time network monitoring and ML-based anomaly detection
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from datetime import datetime
import logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import custom modules
from app.core.config import settings
from app.core.database import engine, get_db
from app.core.security import get_current_user
from app.api import network, anomaly, alerts, upload, models as api_models
from app.services.network_monitor import NetworkMonitor
from app.services.anomaly_detector import AnomalyDetector
from app.services.alert_manager import AlertManager
from app.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global service instances
network_monitor = None
anomaly_detector = None
alert_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Network Anomaly Detection System...")
    
    # Initialize services
    global network_monitor, anomaly_detector, alert_manager
    network_monitor = NetworkMonitor()
    anomaly_detector = AnomalyDetector()
    alert_manager = AlertManager()
    
    # Start background monitoring
    asyncio.create_task(background_monitoring())
    
    logger.info("System started successfully!")
    yield
    
    # Shutdown
    logger.info("Shutting down system...")
    if network_monitor:
        await network_monitor.stop()

# Create FastAPI app
app = FastAPI(
    title="Network Anomaly Detection API",
    description="Real-time network security monitoring with ML-based anomaly detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(network.router, prefix="/api/v1/network", tags=["network"])
app.include_router(anomaly.router, prefix="/api/v1/anomaly", tags=["anomaly"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["alerts"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(api_models.router, prefix="/api/v1/models", tags=["models"])

# Mount static files
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Network Anomaly Detection System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        # db = next(get_db())
        # db.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "connected",
                "network_monitor": "running" if network_monitor else "stopped",
                "anomaly_detector": "ready" if anomaly_detector else "not_ready",
                "alert_manager": "active" if alert_manager else "inactive"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/v1/status")
async def system_status():
    """Get system status"""
    return {
        "monitoring_active": network_monitor.is_active() if network_monitor else False,
        "models_loaded": anomaly_detector.is_ready() if anomaly_detector else False,
        "alerts_count": await alert_manager.get_alert_count() if alert_manager else 0,
        "uptime": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

async def background_monitoring():
    """Background task for continuous network monitoring"""
    while True:
        try:
            if network_monitor and network_monitor.is_active():
                # Collect network data
                network_data = await network_monitor.collect_data()
                
                if network_data:
                    # Detect anomalies
                    anomalies = await anomaly_detector.detect(network_data)
                    
                    # Generate alerts for anomalies
                    if anomalies:
                        await alert_manager.process_anomalies(anomalies)
                
            await asyncio.sleep(5)  # Monitor every 5 seconds
            
        except Exception as e:
            logger.error(f"Background monitoring error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )