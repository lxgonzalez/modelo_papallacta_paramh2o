"""
Inicializador de servicios globales.
"""

from .services.station_service import StationService
from .services.weather_service import WeatherDataService
from .services.prediction_service import PredictionService
from .services.analysis_service import AnalysisService


class ServiceManager:
    """Gestor de servicios de la aplicación"""
    
    def __init__(self):
        self._station_service = None
        self._weather_service = None
        self._prediction_service = None
        self._analysis_service = None
    
    @property
    def station_service(self) -> StationService:
        if self._station_service is None:
            self._station_service = StationService()
        return self._station_service
    
    @property
    def weather_service(self) -> WeatherDataService:
        if self._weather_service is None:
            self._weather_service = WeatherDataService()
        return self._weather_service
    
    @property
    def prediction_service(self) -> PredictionService:
        if self._prediction_service is None:
            self._prediction_service = PredictionService()
        return self._prediction_service
    
    @property
    def analysis_service(self) -> AnalysisService:
        if self._analysis_service is None:
            self._analysis_service = AnalysisService()
        return self._analysis_service
    
    def initialize_all(self):
        """Inicializa todos los servicios"""
        # Cargar modelos de estaciones
        self.station_service.load_all_models()


# Instancia global del gestor de servicios
service_manager = ServiceManager()
