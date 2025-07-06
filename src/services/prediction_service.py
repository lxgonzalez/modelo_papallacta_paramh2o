"""
Servicio para realizar predicciones meteorológicas.
"""

import numpy as np
from typing import List, Dict, Any, Tuple

from ..config.settings import AppConfig, StationConfig
from ..models.data_models import HistoricalWeatherData, PredictionStats
from ..utils.validators import DataCleaner
from ..utils.logging import app_logger, error_handler, PerformanceTimer


class PredictionService:
    """Servicio para realizar predicciones meteorológicas"""
    
    def __init__(self):
        self.logger = app_logger
        self.error_handler = error_handler
        self.time_step = AppConfig.TIME_STEP
        self.feature_columns = StationConfig.FEATURE_COLUMNS
    
    def preprocess_data(self, historical_data: HistoricalWeatherData, scaler) -> np.ndarray:
        """Preprocesa los datos históricos para la predicción"""
        try:
            # Convertir a formato requerido por el modelo
            data_dict = historical_data.to_dict()
            
            # Crear array 2D con las características
            data = np.array([
                data_dict[feature] for feature in self.feature_columns
            ]).T
            
            # Validar datos
            if data.size == 0:
                raise ValueError("No hay datos para procesar")
            
            # Limpiar datos NaN
            data = self._clean_nan_values(data)
            
            # Normalizar con el scaler
            with PerformanceTimer(self.logger, "Normalización de datos"):
                data_scaled = scaler.transform(data)
            
            # Crear secuencias temporales
            return self._create_sequences(data_scaled)
            
        except Exception as e:
            self.error_handler.log_error(e, {'data_shape': data.shape if 'data' in locals() else 'unknown'})
            raise ValueError(f"Error en el preprocesamiento: {str(e)}")
    
    def _clean_nan_values(self, data: np.ndarray) -> np.ndarray:
        """Limpia valores NaN usando interpolación"""
        if np.any(np.isnan(data)):
            self.logger.warning("Valores NaN encontrados, aplicando interpolación")
            
            # Aplicar interpolación por columna
            for col in range(data.shape[1]):
                column_data = data[:, col].tolist()
                interpolated = DataCleaner.interpolate_missing_values(column_data)
                data[:, col] = np.array(interpolated)
        
        return data
    
    def _create_sequences(self, data_scaled: np.ndarray) -> np.ndarray:
        """Crea secuencias temporales para el modelo LSTM"""
        if len(data_scaled) < self.time_step:
            raise ValueError(
                f"Datos insuficientes para crear secuencias. "
                f"Se requieren al menos {self.time_step} puntos, "
                f"pero se obtuvieron {len(data_scaled)}."
            )
        
        X = []
        for i in range(len(data_scaled) - self.time_step + 1):
            X.append(data_scaled[i:i + self.time_step])
        
        return np.array(X)
    
    def predict(self, model, X_test: np.ndarray, scaler) -> np.ndarray:
        """Realiza predicciones usando el modelo LSTM"""
        try:
            with PerformanceTimer(self.logger, "Predicción LSTM"):
                # Realizar predicción
                predictions = model.predict(X_test, verbose=0)
                
                # Desnormalizar predicciones
                predictions_denorm = scaler.inverse_transform(predictions)
                
                self.logger.info(f"Predicciones generadas: {len(predictions_denorm)} registros")
                
                return predictions_denorm
                
        except Exception as e:
            self.error_handler.log_error(e, {
                'X_test_shape': X_test.shape,
                'model_type': type(model).__name__
            })
            raise ValueError(f"Error en la predicción: {str(e)}")
    
    def calculate_prediction_stats(self, predictions: List[List[float]]) -> PredictionStats:
        """Calcula estadísticas de las predicciones"""
        try:
            return PredictionStats.from_predictions(predictions)
        except Exception as e:
            self.error_handler.log_error(e, {'predictions_count': len(predictions)})
            raise ValueError(f"Error calculando estadísticas: {str(e)}")
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """Valida que las predicciones sean válidas"""
        if predictions.size == 0:
            return False
        
        # Verificar que no haya valores infinitos o NaN
        if np.any(np.isinf(predictions)) or np.any(np.isnan(predictions)):
            self.logger.warning("Predicciones contienen valores inválidos")
            return False
        
        # Verificar rangos razonables
        if predictions.shape[1] >= 3:
            temps = predictions[:, 1]  # Temperaturas
            if np.any(temps < -50) or np.any(temps > 60):
                self.logger.warning("Temperaturas fuera del rango razonable")
                return False
        
        return True
