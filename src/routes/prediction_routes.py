"""
Rutas para endpoints de predicciones meteorológicas.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any

from ..models.data_models import WeatherDataRequest, APIResponse
from ..service_manager import service_manager
from ..utils.validators import RequestValidator, ValidationError
from ..utils.logging import app_logger, error_handler, PerformanceTimer

# Crear blueprint
prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal para predicciones meteorológicas con análisis agrícola"""
    try:
        # Validar datos de entrada
        data = request.get_json()
        RequestValidator.validate_json_data(data)
        RequestValidator.validate_required_fields(data, ['date', 'latitude', 'longitude'])
        
        # Crear request model
        weather_request = WeatherDataRequest(
            date=data['date'],
            latitude=float(data['latitude']),
            longitude=float(data['longitude']),
            include_analysis=data.get('include_analysis', True),
            analysis_types=data.get('analysis_types')
        )
        
        app_logger.info(f"Petición de predicción: {weather_request.date}, "
                       f"({weather_request.latitude}, {weather_request.longitude})")
        
        with PerformanceTimer(app_logger, "Predicción completa"):
            # 1. Encontrar estación más cercana
            nearest_station, distance = service_manager.station_service.find_nearest_station(
                weather_request.latitude, weather_request.longitude
            )
            
            # 2. Verificar disponibilidad del modelo
            if not service_manager.station_service.is_model_available(nearest_station):
                available_stations = service_manager.station_service.get_available_stations()
                return jsonify(APIResponse(
                    success=False,
                    error=f'Modelo para la estación más cercana ({nearest_station}) no está disponible',
                    details={
                        'nearest_station': nearest_station,
                        'distance': round(distance, 6),
                        'available_stations': available_stations
                    }
                ).to_dict()), 503
            
            app_logger.info(f"Usando modelo de estación: {nearest_station} (distancia: {distance:.6f})")
            
            # 3. Obtener modelo y scaler
            model, scaler = service_manager.station_service.get_model_and_scaler(nearest_station)
            
            # 4. Obtener datos históricos
            historical_data = service_manager.weather_service.get_historical_data(
                weather_request.date, weather_request.latitude, weather_request.longitude
            )
            
            # 5. Preprocesar datos
            X_test = service_manager.prediction_service.preprocess_data(historical_data, scaler)
            
            # 6. Realizar predicción
            predictions = service_manager.prediction_service.predict(model, X_test, scaler)
            
            # 7. Validar predicciones
            if not service_manager.prediction_service.validate_predictions(predictions):
                return jsonify(APIResponse(
                    success=False,
                    error="Las predicciones generadas no son válidas"
                ).to_dict()), 500
            
            # 8. Análisis con Gemini (opcional)
            analysis = None
            analysis_error = None
            
            if weather_request.include_analysis:
                analysis, analysis_error = service_manager.analysis_service.analyze_predictions(
                    predictions.tolist(), data
                )
                if analysis_error:
                    app_logger.warning(f"Error en análisis: {analysis_error}")
            
            # 9. Preparar respuesta
            station_info = service_manager.station_service.get_station_info(
                nearest_station, weather_request.latitude, weather_request.longitude
            )
            
            response_data = {
                'predictions': predictions.tolist(),
                'model_info': {
                    'selected_station': nearest_station,
                    'station_coordinates': {
                        'lat': station_info.latitude,
                        'lon': station_info.longitude
                    },
                    'distance_to_station': round(distance, 6),
                    'distance_unit': 'grados (lat/lon)'
                },
                'historical_data': historical_data.to_dict(),
                'total_predictions': len(predictions),
                'input_coordinates': {
                    'latitude': weather_request.latitude,
                    'longitude': weather_request.longitude
                },
                'requested_date': weather_request.date,
                'analysis_included': analysis is not None
            }
            
            # Agregar análisis si está disponible
            if analysis:
                response_data['agricultural_analysis'] = analysis
            elif analysis_error:
                response_data['analysis_error'] = analysis_error
            
            return jsonify(APIResponse(success=True, data=response_data).to_dict())
            
    except ValidationError as e:
        app_logger.warning(f"Error de validación: {str(e)}")
        return jsonify(APIResponse(success=False, error=str(e)).to_dict()), 400
        
    except Exception as e:
        error_msg = error_handler.log_error(e, {'endpoint': 'predict'})
        return jsonify(APIResponse(success=False, error=error_msg).to_dict()), 500


@prediction_bp.route('/analysis-options', methods=['GET'])
def get_analysis_options():
    """Endpoint para obtener opciones de análisis disponibles"""
    try:
        options = service_manager.analysis_service.get_analysis_options()
        return jsonify(options)
        
    except Exception as e:
        error_msg = error_handler.log_error(e, {'endpoint': 'analysis_options'})
        return jsonify(APIResponse(success=False, error=error_msg).to_dict()), 500
