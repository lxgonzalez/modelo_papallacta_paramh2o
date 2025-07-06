"""
Servicio para obtener datos meteorológicos de APIs externas.
"""

import requests
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta

from ..config.settings import AppConfig
from ..models.data_models import HistoricalWeatherData
from ..utils.validators import DateValidator, CoordinateValidator
from ..utils.logging import app_logger, error_handler, PerformanceTimer


class WeatherDataService:
    """Servicio para obtener datos meteorológicos históricos"""
    
    def __init__(self):
        self.logger = app_logger
        self.error_handler = error_handler
        self.base_url = AppConfig.OPEN_METEO_BASE_URL
        self.timeout = AppConfig.OPEN_METEO_TIMEOUT
    
    def get_historical_data(self, date_str: str, lat: float, lon: float) -> HistoricalWeatherData:
        """Obtiene datos meteorológicos históricos de Open-Meteo"""
        # Validar entrada
        DateValidator.validate_date_string(date_str)
        CoordinateValidator.validate_coordinates(lat, lon)
        
        # Calcular fechas
        end_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = end_date - timedelta(days=AppConfig.HISTORICAL_DAYS)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        self.logger.info(f"Consultando datos históricos desde {start_date_str} hasta {end_date_str}")
        self.logger.info(f"Coordenadas: lat={lat}, lon={lon}")
        
        with PerformanceTimer(self.logger, f"Consulta API Open-Meteo"):
            try:
                data = self._fetch_data_from_api(lat, lon, start_date_str, end_date_str)
                return self._process_api_response(data, start_date_str, end_date_str)
            except Exception as e:
                self.error_handler.log_error(e, {
                    'coordinates': f"({lat}, {lon})",
                    'date_range': f"{start_date_str} to {end_date_str}"
                })
                raise
    
    def _fetch_data_from_api(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict[str, Any]:
        """Realiza la petición a la API de Open-Meteo"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,relative_humidity_2m",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "timezone": "auto"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            self._validate_api_response(data)
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error de conexión con Open-Meteo: {str(e)}")
    
    def _validate_api_response(self, data: Dict[str, Any]) -> None:
        """Valida la respuesta de la API"""
        if 'hourly' not in data:
            raise ValueError("No se encontraron datos horarios en la respuesta de la API")
        
        hourly_data = data['hourly']
        required_keys = ['temperature_2m', 'precipitation', 'relative_humidity_2m']
        
        for key in required_keys:
            if key not in hourly_data:
                raise ValueError(f"Clave '{key}' no encontrada en los datos de la API")
    
    def _process_api_response(self, data: Dict[str, Any], start_date: str, end_date: str) -> HistoricalWeatherData:
        """Procesa la respuesta de la API y crea el objeto HistoricalWeatherData"""
        hourly_data = data['hourly']
        
        # Extraer datos
        temperatures = hourly_data['temperature_2m']
        precipitations = hourly_data['precipitation']
        humidity = hourly_data['relative_humidity_2m']
        timestamps = hourly_data.get('time', [])
        
        # Limpiar datos (remover valores nulos)
        temperatures = self._clean_numeric_data(temperatures)
        precipitations = self._clean_numeric_data(precipitations)
        humidity = self._clean_numeric_data(humidity)
        
        self.logger.info(f"Datos procesados: {len(temperatures)} registros")
        
        return HistoricalWeatherData(
            temperatures=temperatures,
            precipitations=precipitations,
            humidity=humidity,
            timestamps=timestamps,
            start_date=start_date,
            end_date=end_date
        )
    
    def _clean_numeric_data(self, data: List[Any]) -> List[float]:
        """Limpia datos numéricos, reemplazando valores nulos"""
        cleaned = []
        for item in data:
            if item is None:
                cleaned.append(0.0)  # Reemplazar None con 0
            else:
                try:
                    cleaned.append(float(item))
                except (ValueError, TypeError):
                    cleaned.append(0.0)  # Si no se puede convertir, usar 0
        return cleaned
