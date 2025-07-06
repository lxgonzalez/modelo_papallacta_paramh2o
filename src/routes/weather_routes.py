"""
Rutas para endpoints de datos meteorológicos.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any

from ..models.data_models import WeatherDataRequest, APIResponse
from ..service_manager import service_manager
from ..utils.validators import RequestValidator, ValidationError
from ..utils.logging import app_logger, error_handler

# Crear blueprint
weather_bp = Blueprint('weather', __name__)


@weather_bp.route('/weather_data', methods=['POST'])
def get_weather_data():
    """Endpoint para obtener datos meteorológicos históricos"""
    try:
        # Validar datos de entrada
        data = request.get_json()
        RequestValidator.validate_json_data(data)
        RequestValidator.validate_required_fields(data, ['date', 'latitude', 'longitude'])
        
        # Crear request model
        weather_request = WeatherDataRequest(
            date=data['date'],
            latitude=float(data['latitude']),
            longitude=float(data['longitude'])
        )
        
        app_logger.info(f"Petición de datos meteorológicos: {weather_request.date}, "
                       f"({weather_request.latitude}, {weather_request.longitude})")
        
        # Obtener datos históricos
        historical_data = service_manager.weather_service.get_historical_data(
            weather_request.date, 
            weather_request.latitude, 
            weather_request.longitude
        )
        
        # Crear respuesta
        response = APIResponse(
            success=True,
            data={
                'data': historical_data.to_dict(),
                'total_records': len(historical_data.temperatures),
                'coordinates': {
                    'latitude': weather_request.latitude,
                    'longitude': weather_request.longitude
                },
                'requested_date': weather_request.date
            }
        )
        
        return jsonify(response.to_dict())
        
    except ValidationError as e:
        app_logger.warning(f"Error de validación: {str(e)}")
        return jsonify(APIResponse(success=False, error=str(e)).to_dict()), 400
        
    except Exception as e:
        error_msg = error_handler.log_error(e, {'endpoint': 'weather_data'})
        return jsonify(APIResponse(success=False, error=error_msg).to_dict()), 500
