"""
Configuration management for Network Anomaly Detection System
"""

import os
from typing import Optional
from pydantic import BaseSettings, validator
import secrets

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Paths
    MODEL_PATH: str = "./models"
    DATA_PATH: str = "./data"
    LOG_PATH: str = "./logs"
    UPLOAD_PATH: str = "./uploads"
    
    # ML Model Settings
    ISOLATION_FOREST_CONTAMINATION: float = 0.1
    ISOLATION_FOREST_N_ESTIMATORS: int = 100
    AUTOENCODER_ENCODING_DIM: int = 32
    AUTOENCODER_EPOCHS: int = 100
    AUTOENCODER_BATCH_SIZE: int = 32
    ENSEMBLE_IF_WEIGHT: float = 0.6
    ENSEMBLE_AE_WEIGHT: float = 0.4
    
    # Monitoring Settings
    MONITORING_INTERVAL: int = 5  # seconds
    DATA_RETENTION_DAYS: int = 30
    MAX_ALERTS_PER_HOUR: int = 100
    
    # Network Collection Settings
    NETSTAT_ENABLED: bool = True
    PCAP_ENABLED: bool = True
    LOG_PARSING_ENABLED: bool = True
    
    # Prometheus
    PROMETHEUS_PORT: int = 8001
    METRICS_ENABLED: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Feature Engineering
    WINDOW_SIZE_MINUTES: int = 5
    FEATURE_UPDATE_INTERVAL: int = 60  # seconds
    
    # Alerting
    ALERT_THRESHOLD: float = 0.7
    ALERT_COOLDOWN_SECONDS: int = 300
    EMAIL_ALERTS_ENABLED: bool = False
    SLACK_ALERTS_ENABLED: bool = False
    
    @validator("MODEL_PATH", "DATA_PATH", "LOG_PATH", "UPLOAD_PATH")
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        os.makedirs(v, exist_ok=True)
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Validate database URL format"""
        if v and not v.startswith(('postgresql://', 'sqlite:///')):
            raise ValueError("DATABASE_URL must start with 'postgresql://' or 'sqlite:///'")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Feature extraction configuration
FEATURE_CONFIG = {
    "network_features": [
        "src_ip_encoded",
        "dst_ip_encoded", 
        "src_port",
        "dst_port",
        "protocol_encoded",
        "packet_count",
        "byte_count",
        "duration",
        "packets_per_second",
        "bytes_per_second",
        "unique_ports",
        "connection_rate"
    ],
    
    "temporal_features": [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "time_since_last_connection"
    ],
    
    "statistical_features": [
        "packet_size_mean",
        "packet_size_std",
        "inter_arrival_time_mean",
        "inter_arrival_time_std",
        "flow_duration_percentile_95"
    ]
}

# Model training configuration
MODEL_CONFIG = {
    "isolation_forest": {
        "contamination": settings.ISOLATION_FOREST_CONTAMINATION,
        "n_estimators": settings.ISOLATION_FOREST_N_ESTIMATORS,
        "max_samples": "auto",
        "max_features": 1.0,
        "bootstrap": False,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0
    },
    
    "autoencoder": {
        "encoding_dim": settings.AUTOENCODER_ENCODING_DIM,
        "epochs": settings.AUTOENCODER_EPOCHS,
        "batch_size": settings.AUTOENCODER_BATCH_SIZE,
        "validation_split": 0.1,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "mse",
        "hidden_layers": [64, 32, 16],
        "activation": "relu",
        "output_activation": "sigmoid"
    },
    
    "ensemble": {
        "if_weight": settings.ENSEMBLE_IF_WEIGHT,
        "ae_weight": settings.ENSEMBLE_AE_WEIGHT,
        "voting": "soft",
        "threshold_percentile": 95
    }
}

# Alert severity mapping
ALERT_SEVERITY_CONFIG = {
    "low": {
        "threshold_min": 0.3,
        "threshold_max": 0.5,
        "color": "#FFA500",
        "priority": 1
    },
    "medium": {
        "threshold_min": 0.5,
        "threshold_max": 0.7,
        "color": "#FF8C00",
        "priority": 2
    },
    "high": {
        "threshold_min": 0.7,
        "threshold_max": 0.9,
        "color": "#FF4500",
        "priority": 3
    },
    "critical": {
        "threshold_min": 0.9,
        "threshold_max": 1.0,
        "color": "#FF0000",
        "priority": 4
    }
}

# Network protocol mappings
PROTOCOL_MAPPINGS = {
    "tcp": 6,
    "udp": 17,
    "icmp": 1,
    "http": 80,
    "https": 443,
    "ftp": 21,
    "ssh": 22,
    "dns": 53,
    "smtp": 25,
    "pop3": 110,
    "imap": 143
}

# Common attack patterns for classification
ATTACK_PATTERNS = {
    "port_scan": {
        "description": "Port scanning activity detected",
        "indicators": ["multiple_ports", "syn_flood", "connection_attempts"]
    },
    "ddos": {
        "description": "Distributed Denial of Service attack",
        "indicators": ["high_connection_rate", "bandwidth_spike", "multiple_sources"]
    },
    "brute_force": {
        "description": "Brute force login attempt",
        "indicators": ["repeated_failed_auth", "dictionary_attack", "credential_stuffing"]
    },
    "data_exfiltration": {
        "description": "Potential data exfiltration",
        "indicators": ["large_upload", "unusual_timing", "encrypted_tunnel"]
    },
    "malware_communication": {
        "description": "Malware command and control communication",
        "indicators": ["beacon_traffic", "domain_generation", "suspicious_user_agent"]
    }
}