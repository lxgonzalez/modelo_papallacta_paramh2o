"""
Utilidades para validaciones y funciones auxiliares.
"""

from typing import List, Dict, Any
from datetime import datetime
import math


class ValidationError(Exception):
    """Excepción personalizada para errores de validación"""
    pass


class CoordinateValidator:
    """Validador de coordenadas geográficas"""
    
    @staticmethod
    def validate_latitude(latitude: float) -> bool:
        """Valida que la latitud esté en el rango correcto"""
        return -90 <= latitude <= 90
    
    @staticmethod
    def validate_longitude(longitude: float) -> bool:
        """Valida que la longitud esté en el rango correcto"""
        return -180 <= longitude <= 180
    
    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> None:
        """Valida ambas coordenadas y lanza excepción si son inválidas"""
        if not CoordinateValidator.validate_latitude(latitude):
            raise ValidationError(f"Latitud inválida: {latitude}. Debe estar entre -90 y 90")
        if not CoordinateValidator.validate_longitude(longitude):
            raise ValidationError(f"Longitud inválida: {longitude}. Debe estar entre -180 y 180")


class DateValidator:
    """Validador de fechas"""
    
    @staticmethod
    def validate_date_format(date_str: str, format_str: str = '%Y-%m-%d') -> bool:
        """Valida el formato de fecha"""
        try:
            datetime.strptime(date_str, format_str)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_date_string(date_str: str) -> None:
        """Valida formato de fecha y lanza excepción si es inválido"""
        if not DateValidator.validate_date_format(date_str):
            raise ValidationError(f"Formato de fecha inválido: {date_str}. Usar YYYY-MM-DD")


class GeographicCalculator:
    """Calculadora de operaciones geográficas"""
    
    @staticmethod
    def calculate_euclidean_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula la distancia euclidiana entre dos puntos geográficos"""
        return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    
    @staticmethod
    def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calcula la distancia haversine entre dos puntos geográficos en kilómetros
        Más precisa que la distancia euclidiana para coordenadas geográficas
        """
        # Convertir grados a radianes
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Diferencias
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Fórmula haversine
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radio de la Tierra en kilómetros
        earth_radius = 6371
        
        return earth_radius * c


class RequestValidator:
    """Validador de peticiones HTTP"""
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
        """Valida que todos los campos requeridos estén presentes"""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(f"Campos requeridos faltantes: {', '.join(missing_fields)}")
    
    @staticmethod
    def validate_json_data(data: Any) -> None:
        """Valida que los datos sean un diccionario válido"""
        if not data or not isinstance(data, dict):
            raise ValidationError("Se requieren datos JSON válidos")


class ListValidator:
    """Validador de listas"""
    
    @staticmethod
    def validate_list_items(items: List[str], valid_items: List[str]) -> None:
        """Valida que todos los elementos de la lista sean válidos"""
        invalid_items = [item for item in items if item not in valid_items]
        if invalid_items:
            raise ValidationError(f"Elementos inválidos: {', '.join(invalid_items)}. "
                                f"Elementos válidos: {', '.join(valid_items)}")


class DataCleaner:
    """Utilidades para limpieza de datos"""
    
    @staticmethod
    def clean_numeric_list(data: List[Any]) -> List[float]:
        """Limpia una lista de datos numéricos, removiendo valores nulos"""
        cleaned = []
        for item in data:
            if item is not None:
                try:
                    cleaned.append(float(item))
                except (ValueError, TypeError):
                    # Si no se puede convertir, omitir el valor
                    continue
        return cleaned
    
    @staticmethod
    def interpolate_missing_values(data: List[float]) -> List[float]:
        """Interpola valores faltantes en una lista usando interpolación lineal"""
        if not data:
            return data
        
        # Encontrar índices de valores válidos
        valid_indices = [i for i, val in enumerate(data) if val is not None and not math.isnan(val)]
        
        if len(valid_indices) < 2:
            return data
        
        # Interpolar valores faltantes
        result = data.copy()
        for i in range(len(data)):
            if i not in valid_indices:
                # Encontrar el valor anterior y posterior más cercano
                prev_idx = None
                next_idx = None
                
                for idx in valid_indices:
                    if idx < i:
                        prev_idx = idx
                    elif idx > i and next_idx is None:
                        next_idx = idx
                        break
                
                if prev_idx is not None and next_idx is not None:
                    # Interpolación lineal
                    x0, y0 = prev_idx, data[prev_idx]
                    x1, y1 = next_idx, data[next_idx]
                    result[i] = y0 + (y1 - y0) * (i - x0) / (x1 - x0)
                elif prev_idx is not None:
                    # Usar el valor anterior
                    result[i] = data[prev_idx]
                elif next_idx is not None:
                    # Usar el valor siguiente
                    result[i] = data[next_idx]
        
        return result
