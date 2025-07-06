"""
Utilidades para logging y manejo de errores.
"""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime


class Logger:
    """Configurador de logging para la aplicación"""
    
    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Configura un logger con formato estándar"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Evitar duplicar handlers
        if logger.handlers:
            return logger
        
        # Crear handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Crear formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Agregar handler al logger
        logger.addHandler(console_handler)
        
        return logger


class ErrorHandler:
    """Manejador de errores centralizado"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
        """Registra un error y devuelve un mensaje apropiado"""
        error_msg = str(error)
        
        if context:
            context_str = ', '.join([f"{k}: {v}" for k, v in context.items()])
            self.logger.error(f"Error: {error_msg} | Context: {context_str}")
        else:
            self.logger.error(f"Error: {error_msg}")
        
        return error_msg
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Registra una advertencia"""
        if context:
            context_str = ', '.join([f"{k}: {v}" for k, v in context.items()])
            self.logger.warning(f"Warning: {message} | Context: {context_str}")
        else:
            self.logger.warning(f"Warning: {message}")
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Registra información"""
        if context:
            context_str = ', '.join([f"{k}: {v}" for k, v in context.items()])
            self.logger.info(f"Info: {message} | Context: {context_str}")
        else:
            self.logger.info(f"Info: {message}")


class PerformanceTimer:
    """Utilidad para medir tiempo de ejecución"""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Iniciando operación: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"Operación {self.operation_name} completada en {duration:.2f} segundos")


# Crear logger principal para la aplicación
app_logger = Logger.setup_logger('weather_api')
error_handler = ErrorHandler(app_logger)
