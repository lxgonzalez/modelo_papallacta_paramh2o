"""
Configuración de la aplicación.
Centraliza todas las configuraciones y constantes.
"""

import os
from typing import Dict, Any


class AppConfig:
    """Configuración principal de la aplicación"""
    
    # Configuración del servidor
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    # Configuración de APIs externas
    OPEN_METEO_BASE_URL = 'https://archive-api.open-meteo.com/v1/archive'
    OPEN_METEO_TIMEOUT = 30
    
    # Configuración de modelos
    MODEL_BASE_PATH = 'modelo_final_lstm/modelos'
    TIME_STEP = 30
    HISTORICAL_DAYS = 30
    
    # Configuración de Gemini
    GEMINI_MODEL = 'gemini-1.5-flash'
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    @classmethod
    def is_gemini_available(cls) -> bool:
        """Verifica si la API de Gemini está disponible"""
        return bool(cls.GEMINI_API_KEY)


class StationConfig:
    """Configuración de estaciones meteorológicas"""
    
    STATIONS_COORDINATES = {
        'M5023': {'lat': -0.3798, 'lon': -78.1959},
        'M5025': {'lat': -0.3337, 'lon': -78.1985},
        'P34': {'lat': -0.3809, 'lon': -78.1411},
        'P63': {'lat': -0.3206, 'lon': -78.1917}
    }
    
    FEATURE_COLUMNS = [
        'Precipitacion (mm)', 
        'Temperatura (°C)', 
        'Humedad_Relativa (%)'
    ]
    
    @classmethod
    def get_station_names(cls) -> list:
        """Obtiene la lista de nombres de estaciones"""
        return list(cls.STATIONS_COORDINATES.keys())
    
    @classmethod
    def get_station_coordinates(cls, station_name: str) -> Dict[str, float]:
        """Obtiene las coordenadas de una estación específica"""
        return cls.STATIONS_COORDINATES.get(station_name, {})


class AnalysisConfig:
    """Configuración de análisis agrícola"""
    
    ANALYSIS_OPTIONS = {
        'general': {
            'name': 'Condiciones Generales',
            'description': 'Evaluación del clima previsto',
            'key': 'resumen_climatico'
        },
        'cultivos': {
            'name': 'Recomendaciones de Cultivos',
            'description': 'Qué cultivos serían más apropiados',
            'key': 'recomendaciones_cultivos'
        },
        'riego': {
            'name': 'Manejo del Riego',
            'description': 'Recomendaciones basadas en precipitación prevista',
            'key': 'manejo_riego'
        },
        'alertas': {
            'name': 'Alertas Climáticas',
            'description': 'Posibles riesgos climáticos (sequías, exceso de humedad, etc.)',
            'key': 'alertas'
        },
        'cronograma': {
            'name': 'Cronograma Agrícola',
            'description': 'Mejores momentos para siembra, cosecha, etc.',
            'key': 'cronograma_agricola'
        },
        'plagas': {
            'name': 'Manejo de Plagas',
            'description': 'Condiciones que podrían favorecer plagas',
            'key': 'manejo_plagas'
        },
        'suelo': {
            'name': 'Conservación del Suelo',
            'description': 'Medidas preventivas según el clima',
            'key': 'conservacion_suelo'
        }
    }
    
    @classmethod
    def get_analysis_types(cls) -> list:
        """Obtiene la lista de tipos de análisis disponibles"""
        return list(cls.ANALYSIS_OPTIONS.keys())
    
    @classmethod
    def get_analysis_option(cls, analysis_type: str) -> Dict[str, Any]:
        """Obtiene la configuración de un tipo de análisis específico"""
        return cls.ANALYSIS_OPTIONS.get(analysis_type, {})
