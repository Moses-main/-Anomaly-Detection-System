"""
Anomaly Detector Service
ML-based anomaly detection using Isolation Forest and Autoencoder
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Autoencoder will be disabled.")

from app.core.config import settings, MODEL_CONFIG, FEATURE_CONFIG
from app.models.isolation_forest import IsolationForestDetector
from app.models.autoencoder import AutoencoderDetector
from app.models.ensemble import EnsembleDetector
from app.data.processors.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    timestamp: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    model_used: str
    features: Dict[str, Any]
    source_data: Dict[str, Any]
    severity: str
    description: str

class AnomalyDetector:
    """Main anomaly detection service"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        self.model_type = "ensemble"  # isolation_forest, autoencoder, ensemble
        self.threshold = settings.ALERT_THRESHOLD
        
        # Initialize models
        self._initialize_models()
        
        # Load pre-trained models if available
        self._load_models()
        
        logger.info("AnomalyDetector initialized")
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Isolation Forest
            self.models['isolation_forest'] = IsolationForestDetector(
                contamination=MODEL_CONFIG['isolation_forest']['contamination'],
                n_estimators=MODEL_CONFIG['isolation_forest']['n_estimators'],
                random_state=MODEL_CONFIG['isolation_forest']['random_state']
            )
            
            # Autoencoder (if TensorFlow available)
            if TENSORFLOW_AVAILABLE:
                self.models['autoencoder'] = AutoencoderDetector(
                    encoding_dim=MODEL_CONFIG['autoencoder']['encoding_dim'],
                    hidden_layers=MODEL_CONFIG['autoencoder']['hidden_layers']
                )
            
            # Ensemble
            self.models['ensemble'] = EnsembleDetector(
                if_weight=MODEL_CONFIG['ensemble']['if_weight'],
                ae_weight=MODEL_CONFIG['ensemble']['ae_weight']
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            model_path = settings.MODEL_PATH
            
            # Load Isolation Forest
            if_path = os.path.join(model_path, "isolation_forest.pkl")
            if os.path.exists(if_path):
                with open(if_path, 'rb') as f:
                    self.models['isolation_forest'].model = pickle.load(f)
                logger.info("Loaded Isolation Forest model")
            
            # Load Autoencoder
            if TENSORFLOW_AVAILABLE:
                ae_path = os.path.join(model_path, "autoencoder.h5")
                if os.path.exists(ae_path):
                    self.models['autoencoder'].model = keras.models.load_model(ae_path)
                    logger.info("Loaded Autoencoder model")
            
            # Load scalers and encoders
            scaler_path = os.path.join(model_path, "scalers.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info("Loaded feature scalers")
            
            encoder_path = os.path.join(model_path, "encoders.pkl")
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                logger.info("Loaded label encoders")
            
            # Check if models are trained
            self.is_trained = any([
                hasattr(self.models['isolation_forest'], 'model') and self.models['isolation_forest'].model is not None,
                TENSORFLOW_AVAILABLE and hasattr(self.models['autoencoder'], 'model') and self.models['autoencoder'].model is not None
            ])
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_path = settings.MODEL_PATH
            os.makedirs(model_path, exist_ok=True)
            
            # Save Isolation Forest
            if hasattr(self.models['isolation_forest'], 'model') and self.models['isolation_forest'].model:
                with open(os.path.join(model_path, "isolation_forest.pkl"), 'wb') as f:
                    pickle.dump(self.models['isolation_forest'].model, f)
            
            # Save Autoencoder
            if TENSORFLOW_AVAILABLE and hasattr(self.models['autoencoder'], 'model') and self.models['autoencoder'].model:
                self.models['autoencoder'].model.save(os.path.join(model_path, "autoencoder.h5"))
            
            # Save scalers and encoders
            if self.scalers:
                with open(os.path.join(model_path, "scalers.pkl"), 'wb') as f:
                    pickle.dump(self.scalers, f)
            
            if self.encoders:
                with open(os.path.join(model_path, "encoders.pkl"), 'wb') as f:
                    pickle.dump(self.encoders, f)
            
            # Save model metadata
            metadata = {
                'timestamp': datetime.utcnow().isoformat(),
                'model_type': self.model_type,
                'threshold': self.threshold,
                'is_trained': self.is_trained,
                'feature_config': FEATURE_CONFIG,
                'model_config': MODEL_CONFIG
            }
            
            with open(os.path.join(model_path, "metadata.json"), 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def is_ready(self) -> bool:
        """Check if detector is ready for predictions"""
        return self.is_trained
    
    async def train(self, training_data: List[Dict[str, Any]], labels: Optional[List[int]] = None):
        """Train anomaly detection models"""
        try:
            logger.info(f"Training models on {len(training_data)} samples")
            
            # Extract features
            features_df = await self._extract_features(training_data)
            if features_df.empty:
                raise ValueError("No features extracted from training data")
            
            # Prepare features
            X = self._prepare_features(features_df)
            
            # Train Isolation Forest
            logger.info("Training Isolation Forest...")
            self.models['isolation_forest'].fit(X)
            
            # Train Autoencoder (if available)
            if TENSORFLOW_AVAILABLE and 'autoencoder' in self.models:
                logger.info("Training Autoencoder...")
                self.models['autoencoder'].fit(X)
            
            # Train Ensemble
            if 'ensemble' in self.models:
                logger.info("Training Ensemble model...")
                self.models['ensemble'].fit(
                    X, 
                    self.models.get('isolation_forest'),
                    self.models.get('autoencoder')
                )
            
            self.is_trained = True
            self._save_models()
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    async def detect(self, network_data: List[Dict[str, Any]]) -> List[AnomalyResult]:
        """Detect anomalies in network data"""
        if not self.is_trained:
            logger.warning("Models not trained yet, using basic heuristics")
            return await self._heuristic_detection(network_data)
        
        try:
            anomalies = []
            
            # Extract features
            features_df = await self._extract_features(network_data)
            if features_df.empty:
                return []
            
            # Prepare features
            X = self._prepare_features(features_df)
            
            # Get predictions from active model
            if self.model_type == "isolation_forest":
                predictions, scores = self._predict_isolation_forest(X)
            elif self.model_type == "autoencoder" and TENSORFLOW_AVAILABLE:
                predictions, scores = self._predict_autoencoder(X)
            else:  # ensemble
                predictions, scores = self._predict_ensemble(X)
            
            # Process results
            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == 1:  # Anomaly detected
                    severity = self._calculate_severity(score)
                    description = self._generate_description(network_data[i], features_df.iloc[i])
                    
                    anomaly = AnomalyResult(
                        timestamp=network_data[i].get('timestamp', datetime.utcnow().isoformat()),
                        is_anomaly=True,
                        anomaly_score=float(score),
                        confidence=min(abs(score), 1.0),
                        model_used=self.model_type,
                        features=features_df.iloc[i].to_dict(),
                        source_data=network_data[i],
                        severity=severity,
                        description=description
                    )
                    
                    anomalies.append(anomaly)
            
            logger.info(f"Detected {len(anomalies)} anomalies from {len(network_data)} data points")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _predict_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from Isolation Forest"""
        predictions = self.models['isolation_forest'].predict(X)
        scores = self.models['isolation_forest'].decision_function(X)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        predictions = np.where(predictions == -1, 1, 0)
        # Normalize scores to [0, 1]
        scores = np.abs(scores)
        
        return predictions, scores
    
    def _predict_autoencoder(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from Autoencoder"""
        if not TENSORFLOW_AVAILABLE:
            return np.zeros(len(X)), np.zeros(len(X))
        
        predictions, scores = self.models['autoencoder'].predict(X, threshold=self.threshold)
        return predictions, scores
    
    def _predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from Ensemble model"""
        return self.models['ensemble'].predict(X, threshold=self.threshold)
    
    async def _extract_features(self, network_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features from network data"""
        try:
            return await self.feature_extractor.extract(network_data)
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()
    
    def _prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input"""
        try:
            # Select relevant features
            feature_columns = FEATURE_CONFIG.get('network_features', []) + \
                            FEATURE_CONFIG.get('temporal_features', []) + \
                            FEATURE_CONFIG.get('statistical_features', [])
            
            # Use available columns
            available_columns = [col for col in feature_columns if col in features_df.columns]
            if not available_columns:
                available_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
            
            X = features_df[available_columns].fillna(0)
            
            # Scale features
            if 'main_scaler' not in self.scalers:
                self.scalers['main_scaler'] = StandardScaler()
                X_scaled = self.scalers['main_scaler'].fit_transform(X)
            else:
                X_scaled = self.scalers['main_scaler'].transform(X)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([])
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate anomaly severity based on score"""
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_description(self, network_data: Dict[str, Any], features: pd.Series) -> str:
        """Generate human-readable description of anomaly"""
        try:
            descriptions = []
            
            # Check for common anomaly patterns
            src_ip = network_data.get('src_ip', 'unknown')
            dst_ip = network_data.get('dst_ip', 'unknown')
            protocol = network_data.get('protocol', 'unknown')
            src_port = network_data.get('src_port', 0)
            dst_port = network_data.get('dst_port', 0)
            
            # High connection rate
            if features.get('connection_rate', 0) > 100:
                descriptions.append("High connection rate detected")
            
            # Unusual ports
            if dst_port > 50000 or src_port > 50000:
                descriptions.append("Connection to unusual high port")
            
            # Large data transfer
            bytes_total = network_data.get('bytes_sent', 0) + network_data.get('bytes_recv', 0)
            if bytes_total > 1000000:  # 1MB
                descriptions.append("Large data transfer detected")
            
            # Unusual protocol usage
            if protocol.lower() not in ['tcp', 'udp', 'icmp']:
                descriptions.append(f"Unusual protocol: {protocol}")
            
            base_description = f"Anomalous {protocol.upper()} connection from {src_ip}:{src_port} to {dst_ip}:{dst_port}"
            
            if descriptions:
                return f"{base_description}. {'. '.join(descriptions)}"
            else:
                return f"{base_description}. Statistical anomaly detected"
                
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Anomaly detected in network traffic"
    
    async def _heuristic_detection(self, network_data: List[Dict[str, Any]]) -> List[AnomalyResult]:
        """Basic heuristic detection when models are not trained"""
        anomalies = []
        
        try:
            for data in network_data:
                is_anomaly = False
                score = 0.0
                description = "Heuristic detection: "
                
                # Check for suspicious patterns
                dst_port = data.get('dst_port', 0)
                src_port = data.get('src_port', 0)
                protocol = data.get('protocol', '').lower()
                
                # Suspicious ports
                if dst_port in [1234, 1337, 4444, 5555, 31337]:
                    is_anomaly = True
                    score = 0.8
                    description += "Suspicious destination port"
                
                # High ports
                elif dst_port > 60000 or src_port > 60000:
                    is_anomaly = True
                    score = 0.6
                    description += "Very high port number"
                
                # Uncommon protocols
                elif protocol not in ['tcp', 'udp', 'icmp']:
                    is_anomaly = True
                    score = 0.7
                    description += f"Uncommon protocol: {protocol}"
                
                if is_anomaly:
                    anomaly = AnomalyResult(
                        timestamp=data.get('timestamp', datetime.utcnow().isoformat()),
                        is_anomaly=True,
                        anomaly_score=score,
                        confidence=0.5,  # Lower confidence for heuristics
                        model_used="heuristic",
                        features={},
                        source_data=data,
                        severity=self._calculate_severity(score),
                        description=description
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in heuristic detection: {e}")
            return []
    
    def set_model_type(self, model_type: str):
        """Set active model type"""
        if model_type in self.models:
            self.model_type = model_type
            logger.info(f"Switched to {model_type} model")
        else:
            logger.warning(f"Model type {model_type} not available")
    
    def set_threshold(self, threshold: float):
        """Set anomaly detection threshold"""
        self.threshold = max(0.1, min(1.0, threshold))
        logger.info(f"Threshold set to {self.threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'is_trained': self.is_trained,
            'active_model': self.model_type,
            'threshold': self.threshold,
            'available_models': list(self.models.keys()),
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'last_training': getattr(self, 'last_training', None)
        }