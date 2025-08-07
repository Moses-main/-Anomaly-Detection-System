"""
Anomaly Detection API Endpoints
RESTful API for anomaly detection and model management
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pydantic import BaseModel

from app.services.anomaly_detector import AnomalyDetector
from app.services.network_monitor import NetworkMonitor
from app.core.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Global anomaly detector instance
anomaly_detector: Optional[AnomalyDetector] = None

def get_anomaly_detector() -> AnomalyDetector:
    """Dependency to get anomaly detector instance"""
    global anomaly_detector
    if not anomaly_detector:
        anomaly_detector = AnomalyDetector()
    return anomaly_detector

def get_network_monitor() -> NetworkMonitor:
    """Dependency to get network monitor instance"""
    from app.api.network import get_network_monitor
    return get_network_monitor()

class DetectionRequest(BaseModel):
    """Request model for anomaly detection"""
    data: List[Dict[str, Any]]
    model_type: Optional[str] = "ensemble"
    threshold: Optional[float] = 0.7

class TrainingRequest(BaseModel):
    """Request model for model training"""
    training_data: List[Dict[str, Any]]
    labels: Optional[List[int]] = None
    model_type: str