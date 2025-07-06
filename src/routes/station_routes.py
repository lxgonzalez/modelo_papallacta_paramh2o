"""
Rutas para endpoints de estaciones meteorológicas.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any

from ..models.data_models import APIResponse
from ..service_manager import service_manager
from ..utils.validators import RequestValidator, ValidationError, CoordinateValidator
from ..utils.logging import app_logger, error_handler

# Crear blueprint
station_bp = Blueprint('station', __name__)


@station_bp.route('/stations', methods=['GET'])
def get_stations_info():
    """Endpoint para obtener información sobre todas las estaciones"""
    try:
        summary = service_manager.station_service.get_stations_summary()
        
        response_data = {
            'available_stations': {
                station: service_manager.station_service.get_station_info(station, 0, 0).to_dict()
                for station in summary['available_stations']
            },
            'unavailable_stations': {
                station: {'name': station, 'coordinates': summary['coordinates'][station]}
                for station in summary['failed_stations_list']
            },
            'total_stations': summary['total_stations'],
            'models_loaded': summary['loaded_stations']
        }
        
        return jsonify(APIResponse(success=True, data=response_data).to_dict())
        
    except Exception as e:
        error_msg = error_handler.log_error(e, {'endpoint': 'stations'})
        return jsonify(APIResponse(success=False, error=error_msg).to_dict()), 500


@station_bp.route('/nearest_station', methods=['POST'])
def get_nearest_station():
    """Endpoint para encontrar la estación más cercana"""
    try:
        # Validar datos de entrada
        data = request.get_json()
        RequestValidator.validate_json_data(data)
        RequestValidator.validate_required_fields(data, ['latitude', 'longitude'])
        
        # Validar coordenadas
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        CoordinateValidator.validate_coordinates(lat, lon)
        
        app_logger.info(f"Buscando estación más cercana para ({lat}, {lon})")
        
        # Encontrar estación más cercana
        nearest_station, distance = service_manager.station_service.find_nearest_station(lat, lon)
        
        # Obtener información de todas las estaciones
        all_stations_info = service_manager.station_service.get_all_stations_info(lat, lon)
        
        # Preparar respuesta
        response_data = {
            'input_coordinates': {'latitude': lat, 'longitude': lon},
            'nearest_station': all_stations_info[nearest_station].to_dict(),
            'all_stations': {
                name: info.to_dict() for name, info in all_stations_info.items()
            },
            'distance_unit': 'grados (lat/lon)'
        }
        
        return jsonify(APIResponse(success=True, data=response_data).to_dict())
        
    except ValidationError as e:
        app_logger.warning(f"Error de validación: {str(e)}")
        return jsonify(APIResponse(success=False, error=str(e)).to_dict()), 400
        
    except Exception as e:
        error_msg = error_handler.log_error(e, {'endpoint': 'nearest_station'})
        return jsonify(APIResponse(success=False, error=error_msg).to_dict()), 500
