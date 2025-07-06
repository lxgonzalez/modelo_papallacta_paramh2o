"""
Aplicación principal con arquitectura Clean Code.
"""

from flask import Flask
from dotenv import load_dotenv
import os

from src.config.settings import AppConfig
from src.service_manager import service_manager
from src.routes.weather_routes import weather_bp
from src.routes.prediction_routes import prediction_bp
from src.routes.station_routes import station_bp
from src.routes.info_routes import info_bp
from src.utils.logging import app_logger


class WeatherApp:
    """Clase principal de la aplicación"""
    
    def __init__(self):
        self.app = None
        self.logger = app_logger
        
    def create_app(self) -> Flask:
        """Crea y configura la aplicación Flask"""
        # Cargar variables de entorno
        load_dotenv()
        
        # Crear aplicación Flask
        self.app = Flask(__name__)
        
        # Configurar aplicación
        self._configure_app()
        
        # Registrar blueprints
        self._register_blueprints()
        
        # Inicializar servicios
        self._initialize_services()
        
        return self.app
    
    def _configure_app(self):
        """Configura la aplicación Flask"""
        # Configuraciones básicas
        self.app.config['DEBUG'] = AppConfig.DEBUG
        self.app.config['JSON_SORT_KEYS'] = False
        
        # Logging
        self.logger.info("Aplicación Flask configurada")
    
    def _register_blueprints(self):
        """Registra todos los blueprints"""
        blueprints = [
            (info_bp, ''),
            (weather_bp, ''),
            (prediction_bp, ''),
            (station_bp, '')
        ]
        
        for blueprint, url_prefix in blueprints:
            self.app.register_blueprint(blueprint, url_prefix=url_prefix)
            
        self.logger.info(f"Blueprints registrados: {len(blueprints)}")
    
    def _initialize_services(self):
        """Inicializa los servicios necesarios"""
        self.logger.info("Inicializando servicios...")
        
        # Inicializar todos los servicios
        service_manager.initialize_all()
        
        # Obtener información de estaciones
        summary = service_manager.station_service.get_stations_summary()
        
        # Log de estado
        self.logger.info(f"Estaciones cargadas: {summary['available_stations']}")
        if summary['failed_stations_list']:
            self.logger.warning(f"Estaciones fallidas: {summary['failed_stations_list']}")
    
    def run(self):
        """Ejecuta la aplicación"""
        self._log_startup_info()
        
        self.app.run(
            host=AppConfig.HOST,
            port=AppConfig.PORT,
            debug=AppConfig.DEBUG
        )
    
    def _log_startup_info(self):
        """Registra información de inicio"""
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO SERVIDOR DE PREDICCIONES METEOROLÓGICAS")
        self.logger.info("=" * 60)
        
        # Verificar configuración de Gemini
        if AppConfig.is_gemini_available():
            self.logger.info("✓ GEMINI_API_KEY configurada correctamente")
        else:
            self.logger.warning("⚠️  GEMINI_API_KEY no configurada - Análisis agrícola deshabilitado")
        
        # Información de endpoints
        self.logger.info("\nEndpoints disponibles:")
        endpoints = [
            "GET  /              - Información de la API",
            "GET  /health        - Estado del servidor",
            "GET  /stations      - Información de estaciones",
            "GET  /analysis-options - Opciones de análisis agrícola",
            "POST /weather_data  - Datos meteorológicos históricos",
            "POST /predict       - Predicciones LSTM + análisis agrícola",
            "POST /nearest_station - Encontrar estación más cercana"
        ]
        
        for endpoint in endpoints:
            self.logger.info(f"  {endpoint}")
        
        # Información de configuración
        self.logger.info(f"\nConfiguración:")
        self.logger.info(f"  Host: {AppConfig.HOST}")
        self.logger.info(f"  Puerto: {AppConfig.PORT}")
        self.logger.info(f"  Debug: {AppConfig.DEBUG}")
        self.logger.info(f"  Ruta modelos: {AppConfig.MODEL_BASE_PATH}")
        
        # Información de estaciones
        available_stations = service_manager.station_service.get_available_stations()
        self.logger.info(f"  Estaciones disponibles: {available_stations}")
        
        self.logger.info("=" * 60)


def create_app() -> Flask:
    """Factory function para crear la aplicación"""
    weather_app = WeatherApp()
    return weather_app.create_app()


if __name__ == '__main__':
    try:
        weather_app = WeatherApp()
        app = weather_app.create_app()
        weather_app.run()
    except KeyboardInterrupt:
        app_logger.info("Servidor detenido por el usuario")
    except Exception as e:
        app_logger.error(f"Error crítico: {str(e)}")
        raise
