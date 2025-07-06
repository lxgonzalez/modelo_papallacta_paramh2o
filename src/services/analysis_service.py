"""
Servicio para análisis agrícola usando Google Gemini.
"""

import json
import google.generativeai as genai
from typing import Dict, Any, List, Optional, Tuple

from ..config.settings import AppConfig, AnalysisConfig
from ..models.data_models import PredictionStats
from ..utils.logging import app_logger, error_handler, PerformanceTimer


class AnalysisService:
    """Servicio para análisis agrícola con IA"""
    
    def __init__(self):
        self.logger = app_logger
        self.error_handler = error_handler
        self.model_name = AppConfig.GEMINI_MODEL
        self.analysis_options = AnalysisConfig.ANALYSIS_OPTIONS
        
        # Configurar Gemini si está disponible
        if AppConfig.is_gemini_available():
            genai.configure(api_key=AppConfig.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None
    
    def is_available(self) -> bool:
        """Verifica si el servicio de análisis está disponible"""
        return self.model is not None and AppConfig.is_gemini_available()
    
    def analyze_predictions(self, predictions: List[List[float]], location_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Analiza las predicciones meteorológicas para agricultura"""
        if not self.is_available():
            return None, "Servicio de análisis no disponible - API key de Gemini no configurada"
        
        if not predictions:
            return None, "No se encontraron predicciones para analizar"
        
        try:
            with PerformanceTimer(self.logger, "Análisis con Gemini"):
                # Calcular estadísticas
                stats = PredictionStats.from_predictions(predictions)
                
                # Obtener tipos de análisis solicitados
                requested_analyses = location_data.get('analysis_types', AnalysisConfig.get_analysis_types())
                
                # Crear prompt
                prompt = self._create_analysis_prompt(stats, location_data, requested_analyses)
                
                # Generar análisis
                response = self.model.generate_content(prompt)
                
                # Procesar respuesta
                return self._process_analysis_response(response.text)
                
        except Exception as e:
            error_msg = f"Error al analizar con Gemini: {str(e)}"
            self.error_handler.log_error(e, {
                'predictions_count': len(predictions),
                'location': f"({location_data.get('latitude', 'unknown')}, {location_data.get('longitude', 'unknown')})"
            })
            return None, error_msg
    
    def _create_analysis_prompt(self, stats: PredictionStats, location_data: Dict[str, Any], requested_analyses: List[str]) -> str:
        """Crea el prompt para el análisis con Gemini"""
        prompt = f"""
        Analiza las siguientes predicciones meteorológicas para el área agrícola:

        UBICACIÓN:
        - Latitud: {location_data.get('latitude', 'N/A')}
        - Longitud: {location_data.get('longitude', 'N/A')}
        - Fecha inicial: {location_data.get('date', 'N/A')}

        PREDICCIONES METEOROLÓGICAS ({stats.total_predictions} horas hacia el futuro):
        - Precipitación promedio: {stats.avg_precipitation:.2f} mm
        - Temperatura promedio: {stats.avg_temperature:.2f} °C
        - Humedad promedio: {stats.avg_humidity:.2f} %

        ESTADÍSTICAS DETALLADAS:
        - Precipitación: mínima {stats.min_precipitation:.2f} mm, máxima {stats.max_precipitation:.2f} mm
        - Temperatura: mínima {stats.min_temperature:.2f} °C, máxima {stats.max_temperature:.2f} °C
        - Humedad: mínima {stats.min_humidity:.2f} %, máxima {stats.max_humidity:.2f} %

        ANÁLISIS SOLICITADO:
        Proporciona un análisis detallado para agricultura que incluya ÚNICAMENTE los siguientes aspectos:
        """
        
        # Agregar secciones específicas
        analysis_sections = {}
        section_number = 1
        
        for analysis_type in requested_analyses:
            if analysis_type in self.analysis_options:
                option = self.analysis_options[analysis_type]
                prompt += f"\n{section_number}. **{option['name']}**: {option['description']}"
                
                # Definir estructura de respuesta
                if analysis_type == 'cultivos':
                    analysis_sections[option['key']] = ["lista de cultivos recomendados"]
                elif analysis_type == 'alertas':
                    analysis_sections[option['key']] = ["lista de alertas climáticas"]
                else:
                    analysis_sections[option['key']] = "descripción detallada"
                
                section_number += 1
        
        # Estructura JSON
        json_structure = json.dumps(analysis_sections, indent=2, ensure_ascii=False)
        
        prompt += f"""

        FORMATO DE RESPUESTA:
        Responde en formato JSON con la siguiente estructura:
        {json_structure}
        
        IMPORTANTE: 
        - Solo incluye los análisis solicitados en la respuesta JSON
        - Usa el idioma español
        - Sé específico y práctico en las recomendaciones
        - Considera las condiciones climáticas locales de Ecuador/Quito
        """
        
        return prompt
    
    def _process_analysis_response(self, response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Procesa la respuesta de Gemini y extrae el JSON"""
        try:
            # Buscar JSON en la respuesta
            json_text = self._extract_json_from_response(response_text)
            
            if not json_text:
                # Si no se puede extraer JSON, devolver como texto plano
                return {"analisis_texto": response_text}, None
            
            # Parsear JSON
            analysis = json.loads(json_text)
            return analysis, None
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"No se pudo parsear JSON de la respuesta: {str(e)}")
            return {"analisis_texto": response_text}, None
        except Exception as e:
            return None, f"Error procesando respuesta: {str(e)}"
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extrae el JSON de la respuesta de Gemini"""
        # Buscar bloques de código JSON
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            if json_end > json_start:
                return response_text[json_start:json_end].strip()
        
        # Buscar JSON directo
        if '{' in response_text and '}' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_end > json_start:
                return response_text[json_start:json_end]
        
        return None
    
    def get_analysis_options(self) -> Dict[str, Any]:
        """Obtiene las opciones de análisis disponibles"""
        return {
            'status': 'success',
            'message': 'Opciones de análisis disponibles',
            'options': self.analysis_options,
            'gemini_available': self.is_available(),
            'usage': {
                'description': 'Incluye "analysis_types" en tu petición para especificar qué análisis deseas',
                'example': {
                    'analysis_types': ['general', 'cultivos', 'riego'],
                    'include_analysis': True
                }
            }
        }
