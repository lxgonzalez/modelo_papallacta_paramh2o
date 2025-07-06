"""
Rutas para endpoints de información general y salud.
"""

from flask import Blueprint, jsonify
from datetime import datetime

from ..models.data_models import APIResponse
from ..service_manager import service_manager
from ..config.settings import AppConfig, StationConfig
from ..utils.logging import app_logger, error_handler

# Crear blueprint
info_bp = Blueprint('info', __name__)


@info_bp.route('/', methods=['GET'])
def api_info():
    """Endpoint de información general de la API"""
    try:
        available_stations = service_manager.station_service.get_available_stations()
        
        response_data = {
            'message': 'API de Datos Meteorológicos y Predicciones LSTM',
            'description': 'Selecciona automáticamente el modelo de la subestación más cercana',
            'version': '2.0.0',
            'available_stations': available_stations,
            'station_coordinates': StationConfig.STATIONS_COORDINATES,
            'endpoints': {
                '/weather_data': 'POST - Obtener datos meteorológicos históricos (últimos 30 días)',
                '/predict': 'POST - Realizar predicciones usando modelo LSTM + análisis agrícola con Gemini',
                '/nearest_station': 'POST - Encontrar la estación más cercana a unas coordenadas',
                '/stations': 'GET - Información sobre todas las estaciones disponibles',
                '/analysis-options': 'GET - Opciones de análisis agrícola disponibles',
                '/health': 'GET - Verificar estado del servidor'
            },
            'formato_request': {
                'date': 'YYYY-MM-DD (fecha de referencia)',
                'latitude': 'número decimal (coordenada Y)',
                'longitude': 'número decimal (coordenada X)',
                'include_analysis': 'booleano (opcional, por defecto true)',
                'analysis_types': 'lista de strings (opcional, ver /analysis-options)'
            },
            'ejemplo_quito': {
                'date': '2024-06-01',
                'latitude': -0.35,
                'longitude': -78.17,
                'include_analysis': True,
                'analysis_types': ['general', 'cultivos', 'riego']
            },
            'gemini_status': {
                'available': service_manager.analysis_service.is_available(),
                'required_env_var': 'GEMINI_API_KEY'
            }
        }
        
        return jsonify(APIResponse(success=True, data=response_data).to_dict())
        
    except Exception as e:
        error_msg = error_handler.log_error(e, {'endpoint': 'info'})
        return jsonify(APIResponse(success=False, error=error_msg).to_dict()), 500


@info_bp.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud del servidor"""
    try:
        summary = service_manager.station_service.get_stations_summary()
        
        response_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'station_service': {
                    'available_stations': summary['available_stations'],
                    'total_models_loaded': summary['loaded_stations'],
                    'failed_stations': summary['failed_stations']
                },
                'analysis_service': {
                    'available': service_manager.analysis_service.is_available(),
                    'gemini_configured': AppConfig.is_gemini_available()
                }
            },
            'station_coordinates': summary['coordinates'],
            'configuration': {
                'model_path': AppConfig.MODEL_BASE_PATH,
                'time_step': AppConfig.TIME_STEP,
                'historical_days': AppConfig.HISTORICAL_DAYS
            }
        }
        
        return jsonify(APIResponse(success=True, data=response_data).to_dict())
        
    except Exception as e:
        error_msg = error_handler.log_error(e, {'endpoint': 'health'})
        return jsonify(APIResponse(success=False, error=error_msg).to_dict()), 500
