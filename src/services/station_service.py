"""
Servicio para manejo de estaciones meteorológicas.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional
from tensorflow.keras.models import load_model

from ..config.settings import StationConfig, AppConfig
from ..models.data_models import StationInfo
from ..utils.validators import GeographicCalculator
from ..utils.logging import app_logger, error_handler


class StationService:
    """Servicio para gestión de estaciones meteorológicas"""
    
    def __init__(self):
        self.logger = app_logger
        self.error_handler = error_handler
        self.models: Dict[str, any] = {}
        self.scalers: Dict[str, any] = {}
        self.loaded_stations: List[str] = []
        self.failed_stations: List[str] = []
        
    def load_all_models(self) -> Tuple[List[str], List[str]]:
        """Carga todos los modelos y scalers disponibles"""
        self.logger.info("Iniciando carga de modelos LSTM...")
        
        for station_name in StationConfig.get_station_names():
            try:
                model = self._load_lstm_model(station_name)
                scaler = self._load_scaler(station_name)
                
                self.models[station_name] = model
                self.scalers[station_name] = scaler
                self.loaded_stations.append(station_name)
                
                self.logger.info(f"✓ Modelo y scaler {station_name} cargados correctamente")
                
            except Exception as e:
                self.failed_stations.append(station_name)
                self.error_handler.log_error(e, {'station': station_name})
        
        self.logger.info(f"Carga completada. Estaciones cargadas: {len(self.loaded_stations)}")
        return self.loaded_stations, self.failed_stations
    
    def _load_lstm_model(self, station_name: str):
        """Carga un modelo LSTM específico"""
        model_path = os.path.join(AppConfig.MODEL_BASE_PATH, f'{station_name}_lstm_model.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        
        return load_model(model_path)
    
    def _load_scaler(self, station_name: str):
        """Carga un scaler específico"""
        scaler_path = os.path.join(AppConfig.MODEL_BASE_PATH, f'{station_name}_scaler.pkl')
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"No se encontró el scaler: {scaler_path}")
        
        return joblib.load(scaler_path)
    
    def find_nearest_station(self, target_lat: float, target_lon: float) -> Tuple[str, float]:
        """Encuentra la estación más cercana a las coordenadas dadas"""
        min_distance = float('inf')
        nearest_station = None
        
        for station_name, coords in StationConfig.STATIONS_COORDINATES.items():
            distance = GeographicCalculator.calculate_euclidean_distance(
                target_lat, target_lon, coords['lat'], coords['lon']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_station = station_name
        
        return nearest_station, min_distance
    
    def get_station_info(self, station_name: str, target_lat: float, target_lon: float) -> StationInfo:
        """Obtiene información completa de una estación"""
        coords = StationConfig.get_station_coordinates(station_name)
        
        if not coords:
            raise ValueError(f"Estación {station_name} no encontrada")
        
        distance = GeographicCalculator.calculate_euclidean_distance(
            target_lat, target_lon, coords['lat'], coords['lon']
        )
        
        return StationInfo(
            name=station_name,
            latitude=coords['lat'],
            longitude=coords['lon'],
            distance=distance,
            model_available=self.is_model_available(station_name)
        )
    
    def get_all_stations_info(self, target_lat: float, target_lon: float) -> Dict[str, StationInfo]:
        """Obtiene información de todas las estaciones"""
        stations_info = {}
        
        for station_name in StationConfig.get_station_names():
            try:
                stations_info[station_name] = self.get_station_info(station_name, target_lat, target_lon)
            except Exception as e:
                self.error_handler.log_error(e, {'station': station_name})
        
        return stations_info
    
    def is_model_available(self, station_name: str) -> bool:
        """Verifica si el modelo de una estación está disponible"""
        return station_name in self.models and station_name in self.scalers
    
    def get_model_and_scaler(self, station_name: str) -> Tuple[any, any]:
        """Obtiene el modelo y scaler de una estación"""
        if not self.is_model_available(station_name):
            raise ValueError(f"Modelo para la estación {station_name} no está disponible")
        
        return self.models[station_name], self.scalers[station_name]
    
    def get_available_stations(self) -> List[str]:
        """Obtiene la lista de estaciones con modelos disponibles"""
        return [station for station in self.loaded_stations if self.is_model_available(station)]
    
    def get_stations_summary(self) -> Dict[str, any]:
        """Obtiene un resumen del estado de todas las estaciones"""
        return {
            'total_stations': len(StationConfig.get_station_names()),
            'loaded_stations': len(self.loaded_stations),
            'failed_stations': len(self.failed_stations),
            'available_stations': self.get_available_stations(),
            'failed_stations_list': self.failed_stations,
            'coordinates': StationConfig.STATIONS_COORDINATES
        }
