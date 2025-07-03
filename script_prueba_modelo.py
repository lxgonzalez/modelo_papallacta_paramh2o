import json
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import os

# Crear la app Flask
app = Flask(__name__)

# Coordenadas de las subestaciones
STATIONS_COORDINATES = {
    'M5023': {'lat': -0.3798, 'lon': -78.1959},
    'M5025': {'lat': -0.3337, 'lon': -78.1985},
    'P34': {'lat': -0.3809, 'lon': -78.1411},
    'P63': {'lat': -0.3206, 'lon': -78.1917}
}

# Función para calcular distancia euclidiana entre dos coordenadas
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia euclidiana entre dos puntos geográficos"""
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

# Función para encontrar la estación más cercana
def find_nearest_station(target_lat, target_lon):
    """Encuentra la estación más cercana a las coordenadas dadas"""
    min_distance = float('inf')
    nearest_station = None
    
    for station, coords in STATIONS_COORDINATES.items():
        distance = calculate_distance(target_lat, target_lon, coords['lat'], coords['lon'])
        if distance < min_distance:
            min_distance = distance
            nearest_station = station
    
    return nearest_station, min_distance

# Función para cargar el modelo LSTM
def load_lstm_model(station_name):
    model_path = f'modelo_final_lstm/modelos/{station_name}_lstm_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

# Función para cargar el scaler
def load_scaler(station_name):
    scaler_path = f'modelo_final_lstm/modelos/{station_name}_scaler.pkl'
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"No se encontró el scaler: {scaler_path}")

# Función para preprocesar los datos desde JSON
def preprocess_data_from_json(json_data, feature_cols, scaler, time_step=30):
    try:
        data = np.array([json_data[feature] for feature in feature_cols]).T  # Convertir a array de 2D
        
        # Verificar que no haya valores None/null
        if np.any(np.isnan(data)):
            # Reemplazar NaN con interpolación lineal simple
            mask = np.isnan(data)
            data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
        
        data_scaled = scaler.transform(data)  # Normalizar los datos con el scaler cargado
        
        X = []
        if len(data_scaled) >= time_step:  # Verificar que tengamos suficientes datos
            for i in range(len(data_scaled) - time_step + 1):
                X.append(data_scaled[i:i + time_step])
            
            X = np.array(X)  # Convertir la lista a un array de numpy
            return X
        else:
            raise ValueError(f"No hay suficientes datos para crear las secuencias. Se requieren al menos {time_step} puntos de datos, pero solo se obtuvieron {len(data_scaled)}.")
    except Exception as e:
        raise ValueError(f"Error en el preprocesamiento: {str(e)}")

# Función para realizar las predicciones con el modelo LSTM
def predict_lstm(model, X_test, scaler):
    try:
        predictions = model.predict(X_test)
        predictions_denorm = scaler.inverse_transform(predictions)  # Desnormalizar las predicciones
        return predictions_denorm
    except Exception as e:
        raise ValueError(f"Error en la predicción: {str(e)}")

# Función para consultar Open Meteo API y obtener datos históricos
def get_historical_data(date_str, lat, lon):
    try:
        # Convertir la fecha de entrada a un formato adecuado
        end_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = end_date - timedelta(days=30)  # 30 días anteriores a la fecha proporcionada
        
        # Convertir las fechas a formato 'YYYY-MM-DD'
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Consultando datos desde {start_date_str} hasta {end_date_str}")
        print(f"Coordenadas: {lat}, {lon}")

        # URL de Open Meteo para datos históricos - CORREGIDA
        url = 'https://archive-api.open-meteo.com/v1/archive'
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "hourly": "temperature_2m,precipitation,relative_humidity_2m",  # Corregido: precipitation en lugar de precipitation_sum
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "timezone": "auto"
        }

        # Hacer la solicitud a Open Meteo con timeout
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'hourly' not in data:
                raise ValueError("No se encontraron datos horarios en la respuesta de la API")
            
            hourly_data = data['hourly']
            
            # Verificar que las claves existan
            required_keys = ['temperature_2m', 'precipitation', 'relative_humidity_2m']
            for key in required_keys:
                if key not in hourly_data:
                    raise ValueError(f"Clave '{key}' no encontrada en los datos de la API")
            
            # Recopilar los datos en formato adecuado
            temperatures = hourly_data['temperature_2m']
            precipitations = hourly_data['precipitation']
            humidity = hourly_data['relative_humidity_2m']
            timestamps = hourly_data.get('time', [])
            
            print(f"Datos obtenidos: {len(temperatures)} registros")
            
            # Retornar los datos en formato JSON para el procesamiento
            return {
                "Precipitacion (mm)": precipitations,
                "Temperatura (°C)": temperatures,
                "Humedad_Relativa (%)": humidity,
                "timestamps": timestamps,
                "periodo": {
                    "inicio": start_date_str,
                    "fin": end_date_str
                }
            }
        else:
            raise ValueError(f"Error al consultar Open Meteo: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error de conexión con Open Meteo: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error al obtener datos históricos: {str(e)}")

# Cargar todos los modelos y scalers disponibles al iniciar la app
models = {}

scalers = {}

def load_all_models():
    """Carga todos los modelos y scalers disponibles"""
    loaded_stations = []
    failed_stations = []
    
    for station in STATIONS_COORDINATES.keys():
        try:
            models[station] = load_lstm_model(station)
            scalers[station] = load_scaler(station)
            loaded_stations.append(station)
            print(f"✓ Modelo y scaler {station} cargados correctamente")
        except Exception as e:
            failed_stations.append(station)
            print(f"✗ Error cargando modelo/scaler {station}: {e}")
    
    return loaded_stations, failed_stations

# Inicializar todos los modelos disponibles
loaded_stations, failed_stations = load_all_models()
print(f"Modelos cargados: {loaded_stations}")
if failed_stations:
    print(f"Modelos no disponibles: {failed_stations}")


# Inicializar todos los modelos disponibles
print("Iniciando carga de modelos...")
loaded_stations, failed_stations = load_all_models()
print(f"Modelos cargados: {loaded_stations}")
if failed_stations:
    print(f"Modelos no disponibles: {failed_stations}")

# Ruta para obtener solo los datos meteorológicos históricos
@app.route('/weather_data', methods=['POST'])
def get_weather_data():
    try:
        # Obtener la fecha y las coordenadas desde la solicitud
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400
        
        date_str = data.get("date")  # Fecha en formato 'YYYY-MM-DD'
        lat = data.get("latitude")
        lon = data.get("longitude")
        
        # Validar que se proporcionaron todos los datos necesarios
        if not all([date_str, lat, lon]):
            return jsonify({'error': 'Se requieren los campos: date, latitude, longitude'}), 400
        
        # Validar formato de fecha
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({'error': 'Formato de fecha inválido. Use YYYY-MM-DD'}), 400
        
        # Validar coordenadas
        try:
            lat = float(lat)
            lon = float(lon)
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise ValueError("Coordenadas fuera de rango")
        except (ValueError, TypeError):
            return jsonify({'error': 'Coordenadas inválidas'}), 400
        
        # Consultar los últimos 30 días de datos usando Open Meteo
        historical_data = get_historical_data(date_str, lat, lon)
        
        # Devolver los datos meteorológicos
        response = {
            'success': True,
            'data': historical_data,
            'total_records': len(historical_data['Temperatura (°C)']),
            'coordinates': {'latitude': lat, 'longitude': lon},
            'requested_date': date_str
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ruta para predicciones (selecciona automáticamente el modelo más cercano)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener la fecha y las coordenadas desde la solicitud
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400
        
        date_str = data.get("date")  # Fecha en formato 'YYYY-MM-DD'
        lat = data.get("latitude")
        lon = data.get("longitude")
        
        # Validar que se proporcionaron todos los datos necesarios
        if not all([date_str, lat, lon]):
            return jsonify({'error': 'Se requieren los campos: date, latitude, longitude'}), 400
        
        # Validar formato de fecha
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({'error': 'Formato de fecha inválido. Use YYYY-MM-DD'}), 400
        
        # Validar coordenadas
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return jsonify({'error': 'Coordenadas inválidas'}), 400
        
        # Encontrar la estación más cercana
        nearest_station, distance = find_nearest_station(lat, lon)
        
        # Verificar que el modelo de la estación más cercana esté disponible
        if nearest_station not in models or nearest_station not in scalers:
            available_stations = list(set(models.keys()) & set(scalers.keys()))
            return jsonify({
                'error': f'Modelo para la estación más cercana ({nearest_station}) no está disponible',
                'nearest_station': nearest_station,
                'distance': round(distance, 6),
                'available_stations': available_stations,
                'station_coordinates': STATIONS_COORDINATES[nearest_station]
            }), 503
        
        print(f"Usando modelo de estación: {nearest_station} (distancia: {distance:.6f})")
        
        # Obtener el modelo y scaler de la estación más cercana
        model = models[nearest_station]
        scaler = scalers[nearest_station]
        
        # Consultar los últimos 30 días de datos usando Open Meteo
        historical_data = get_historical_data(date_str, lat, lon)
        
        # Preprocesar los datos
        feature_cols = ['Precipitacion (mm)', 'Temperatura (°C)', 'Humedad_Relativa (%)']
        X_test = preprocess_data_from_json(historical_data, feature_cols, scaler, time_step=30)

        # Realizar la predicción con el modelo
        predictions = predict_lstm(model, X_test, scaler)

        # Devolver las predicciones como JSON
        response = {
            'success': True,
            'predictions': predictions.tolist(),  # Convertimos a lista para devolver como JSON
            'model_info': {
                'selected_station': nearest_station,
                'station_coordinates': STATIONS_COORDINATES[nearest_station],
                'distance_to_station': round(distance, 6),
                'distance_unit': 'grados (lat/lon)'
            },
            'historical_data': historical_data,
            'total_predictions': len(predictions),
            'input_coordinates': {'latitude': lat, 'longitude': lon},
            'requested_date': date_str
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ruta de salud para verificar que el servidor esté funcionando
@app.route('/health', methods=['GET'])
def health_check():
    available_models = list(set(models.keys()) & set(scalers.keys()))
    return jsonify({
        'status': 'healthy',
        'available_stations': available_models,
        'total_models_loaded': len(available_models),
        'station_coordinates': STATIONS_COORDINATES,
        'timestamp': datetime.now().isoformat()
    })

# Nuevo endpoint para obtener información sobre estaciones
@app.route('/stations', methods=['GET'])
def get_stations_info():
    """Devuelve información sobre todas las estaciones disponibles"""
    available_models = list(set(models.keys()) & set(scalers.keys()))
    unavailable_models = [station for station in STATIONS_COORDINATES.keys() if station not in available_models]
    
    return jsonify({
        'available_stations': {
            station: STATIONS_COORDINATES[station] for station in available_models
        },
        'unavailable_stations': {
            station: STATIONS_COORDINATES[station] for station in unavailable_models
        },
        'total_stations': len(STATIONS_COORDINATES),
        'models_loaded': len(available_models)
    })

# Nuevo endpoint para encontrar la estación más cercana sin hacer predicción
@app.route('/nearest_station', methods=['POST'])
def get_nearest_station():
    """Encuentra la estación más cercana a las coordenadas dadas"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos JSON'}), 400
        
        lat = data.get("latitude")
        lon = data.get("longitude")
        
        if lat is None or lon is None:
            return jsonify({'error': 'Se requieren los campos: latitude, longitude'}), 400
        
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return jsonify({'error': 'Coordenadas inválidas'}), 400
        
        # Encontrar la estación más cercana
        nearest_station, distance = find_nearest_station(lat, lon)
        
        # Calcular distancias a todas las estaciones
        all_distances = {}
        for station, coords in STATIONS_COORDINATES.items():
            dist = calculate_distance(lat, lon, coords['lat'], coords['lon'])
            all_distances[station] = {
                'distance': round(dist, 6),
                'coordinates': coords,
                'model_available': station in models and station in scalers
            }
        
        return jsonify({
            'success': True,
            'input_coordinates': {'latitude': lat, 'longitude': lon},
            'nearest_station': {
                'name': nearest_station,
                'coordinates': STATIONS_COORDINATES[nearest_station],
                'distance': round(distance, 6),
                'model_available': nearest_station in models and nearest_station in scalers
            },
            'all_stations': all_distances,
            'distance_unit': 'grados (lat/lon)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ruta de información sobre el uso de la API
@app.route('/', methods=['GET'])
def api_info():
    available_models = list(set(models.keys()) & set(scalers.keys()))
    return jsonify({
        'message': 'API de Datos Meteorológicos y Predicciones LSTM',
        'description': 'Selecciona automáticamente el modelo de la subestación más cercana',
        'available_stations': available_models,
        'station_coordinates': STATIONS_COORDINATES,
        'endpoints': {
            '/weather_data': 'POST - Obtener datos meteorológicos históricos (últimos 30 días)',
            '/predict': 'POST - Realizar predicciones usando modelo LSTM de la estación más cercana',
            '/nearest_station': 'POST - Encontrar la estación más cercana a unas coordenadas',
            '/stations': 'GET - Información sobre todas las estaciones disponibles',
            '/health': 'GET - Verificar estado del servidor'
        },
        'formato_request': {
            'date': 'YYYY-MM-DD (fecha de referencia)',
            'latitude': 'número decimal (coordenada Y)',
            'longitude': 'número decimal (coordenada X)'
        },
        'ejemplo_quito': {
            'date': '2024-06-01',
            'latitude': -0.35,
            'longitude': -78.17
        }
    })

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    print("Endpoints disponibles:")
    print("- GET  /              - Información de la API")
    print("- GET  /health        - Estado del servidor")
    print("- GET  /stations      - Información de estaciones")
    print("- POST /weather_data  - Datos meteorológicos históricos")
    print("- POST /predict       - Predicciones LSTM (selección automática de modelo)")
    print("- POST /nearest_station - Encontrar estación más cercana")
    print(f"\nEstaciones disponibles: {list(set(models.keys()) & set(scalers.keys()))}")
    print(f"Coordenadas de estaciones: {STATIONS_COORDINATES}")
    app.run(debug=True, port=5000, host='0.0.0.0')