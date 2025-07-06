"""
Modelos de datos para la aplicación.
Define las estructuras de datos y validaciones.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class WeatherDataRequest:
    """Modelo para peticiones de datos meteorológicos"""
    date: str
    latitude: float
    longitude: float
    include_analysis: bool = True
    analysis_types: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validaciones después de la inicialización"""
        self._validate_date()
        self._validate_coordinates()
    
    def _validate_date(self):
        """Valida el formato de fecha"""
        try:
            datetime.strptime(self.date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Formato de fecha inválido. Usar YYYY-MM-DD")
    
    def _validate_coordinates(self):
        """Valida las coordenadas"""
        if not (-90 <= self.latitude <= 90):
            raise ValueError("Latitud debe estar entre -90 y 90")
        if not (-180 <= self.longitude <= 180):
            raise ValueError("Longitud debe estar entre -180 y 180")


@dataclass
class StationInfo:
    """Modelo para información de estaciones"""
    name: str
    latitude: float
    longitude: float
    distance: float
    model_available: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            'name': self.name,
            'coordinates': {'lat': self.latitude, 'lon': self.longitude},
            'distance': round(self.distance, 6),
            'model_available': self.model_available
        }


@dataclass
class HistoricalWeatherData:
    """Modelo para datos meteorológicos históricos"""
    temperatures: List[float]
    precipitations: List[float]
    humidity: List[float]
    timestamps: List[str]
    start_date: str
    end_date: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para procesamiento"""
        return {
            "Precipitacion (mm)": self.precipitations,
            "Temperatura (°C)": self.temperatures,
            "Humedad_Relativa (%)": self.humidity,
            "timestamps": self.timestamps,
            "periodo": {
                "inicio": self.start_date,
                "fin": self.end_date
            }
        }


@dataclass
class PredictionStats:
    """Modelo para estadísticas de predicciones"""
    total_predictions: int
    avg_precipitation: float
    avg_temperature: float
    avg_humidity: float
    min_precipitation: float
    max_precipitation: float
    min_temperature: float
    max_temperature: float
    min_humidity: float
    max_humidity: float
    
    @classmethod
    def from_predictions(cls, predictions: List[List[float]]) -> 'PredictionStats':
        """Crea estadísticas a partir de predicciones"""
        if not predictions:
            raise ValueError("No hay predicciones para calcular estadísticas")
        
        precipitations = [pred[0] for pred in predictions if len(pred) >= 3]
        temperatures = [pred[1] for pred in predictions if len(pred) >= 3]
        humidities = [pred[2] for pred in predictions if len(pred) >= 3]
        
        return cls(
            total_predictions=len(predictions),
            avg_precipitation=sum(precipitations) / len(precipitations) if precipitations else 0,
            avg_temperature=sum(temperatures) / len(temperatures) if temperatures else 0,
            avg_humidity=sum(humidities) / len(humidities) if humidities else 0,
            min_precipitation=min(precipitations) if precipitations else 0,
            max_precipitation=max(precipitations) if precipitations else 0,
            min_temperature=min(temperatures) if temperatures else 0,
            max_temperature=max(temperatures) if temperatures else 0,
            min_humidity=min(humidities) if humidities else 0,
            max_humidity=max(humidities) if humidities else 0
        )


@dataclass
class APIResponse:
    """Modelo base para respuestas de API"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    details: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        result = {'success': self.success}
        if self.data:
            result.update(self.data)
        if self.error:
            result['error'] = self.error
        if self.details:
            result['details'] = self.details
        return result
