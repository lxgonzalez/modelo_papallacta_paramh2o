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
        Analiza estas predicciones meteorológicas para agricultura de manera CONCISA:

        UBICACIÓN: {location_data.get('latitude', 'N/A')}, {location_data.get('longitude', 'N/A')}
        PERIODO: {stats.total_predictions} horas desde {location_data.get('date', 'N/A')}

        DATOS CLIMÁTICOS:
        - Temperatura: {stats.min_temperature:.1f}°C - {stats.max_temperature:.1f}°C (promedio: {stats.avg_temperature:.1f}°C)
        - Precipitación: {stats.min_precipitation:.1f} - {stats.max_precipitation:.1f} mm (promedio: {stats.avg_precipitation:.2f} mm)
        - Humedad: {stats.min_humidity:.0f} - {stats.max_humidity:.0f}% (promedio: {stats.avg_humidity:.0f}%)

        INSTRUCCIONES:
        """
        
        # Agregar secciones específicas con instrucciones claras
        analysis_sections = {}
        
        for analysis_type in requested_analyses:
            if analysis_type in self.analysis_options:
                option = self.analysis_options[analysis_type]
                
                if analysis_type == 'general':
                    prompt += f"\n- RESUMEN CLIMÁTICO: Describe en 2-3 oraciones las condiciones esperadas"
                    analysis_sections[option['key']] = "descripción breve"
                    
                elif analysis_type == 'cultivos':
                    prompt += f"\n- CULTIVOS RECOMENDADOS: Lista 5-7 cultivos apropiados para estas condiciones"
                    analysis_sections[option['key']] = ["lista de cultivos"]
                    
                elif analysis_type == 'riego':
                    prompt += f"\n- MANEJO DE RIEGO: Recomendación específica en 2-3 oraciones sobre frecuencia y método"
                    analysis_sections[option['key']] = "recomendación breve"
                    
                elif analysis_type == 'alertas':
                    prompt += f"\n- ALERTAS: Lista riesgos específicos (heladas, exceso humedad, etc.)"
                    analysis_sections[option['key']] = ["lista de alertas"]
                    
                elif analysis_type == 'cronograma':
                    prompt += f"\n- CRONOGRAMA: Actividades recomendadas para este periodo"
                    analysis_sections[option['key']] = "cronograma breve"
                    
                elif analysis_type == 'plagas':
                    prompt += f"\n- PLAGAS: Riesgos de plagas en estas condiciones"
                    analysis_sections[option['key']] = "recomendación breve"
                    
                elif analysis_type == 'suelo':
                    prompt += f"\n- SUELO: Medidas de conservación necesarias"
                    analysis_sections[option['key']] = "recomendación breve"
        
        # Estructura JSON
        json_structure = json.dumps(analysis_sections, indent=2, ensure_ascii=False)
        
        prompt += f"""

        EJEMPLOS DE RESPUESTA ESPERADA:
        
        Para "resumen_climatico":
        "Clima frío y húmedo. Riesgo de heladas por temperaturas mínimas. Alta humedad favorece hongos."
        
        Para "manejo_riego":
        "Riego ligero cada 3-4 días. Evitar encharcamiento por alta humedad. Usar goteo preferiblemente."
        
        Para "recomendaciones_cultivos":
        ["Papa", "Cebolla", "Ajo", "Arveja", "Habas"]
        
        Para "alertas":
        ["Riesgo de heladas", "Enfermedades fúngicas", "Exceso de humedad"]

        FORMATO DE RESPUESTA JSON:
        {json_structure}
        
        REGLAS IMPORTANTES:
        1. Responde SOLO en formato JSON válido
        2. Máximo 2-3 oraciones por recomendación
        3. Sé específico y práctico
        4. Considera clima ecuatoriano/andino
        5. Para listas, incluye solo los elementos más relevantes (máximo 7 items)
        6. Evita explicaciones largas o redundantes
        7. Usa frases directas sin conectores innecesarios
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
            
            # Post-procesar para acortar respuestas largas
            analysis = self._optimize_response_length(analysis)
            
            return analysis, None
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"No se pudo parsear JSON de la respuesta: {str(e)}")
            return {"analisis_texto": response_text}, None
        except Exception as e:
            return None, f"Error procesando respuesta: {str(e)}"
    
    def _optimize_response_length(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimiza la longitud de las respuestas para que sean más concisas"""
        optimized = {}
        
        for key, value in analysis.items():
            if isinstance(value, str):
                # Acortar textos largos
                if len(value) > 300:  # Si es muy largo
                    # Tomar solo las primeras 2-3 oraciones
                    sentences = value.split('. ')
                    if len(sentences) > 3:
                        value = '. '.join(sentences[:3]) + '.'
                    
                    # Remover frases redundantes
                    value = self._remove_redundant_phrases(value)
                
                optimized[key] = value
                
            elif isinstance(value, list):
                # Limitar listas a máximo 7 elementos
                if len(value) > 7:
                    value = value[:7]
                optimized[key] = value
            else:
                optimized[key] = value
        
        return optimized
    
    def _remove_redundant_phrases(self, text: str) -> str:
        """Remueve frases redundantes y conectores innecesarios"""
        # Frases a remover o simplificar
        redundant_phrases = [
            "Es importante mencionar que",
            "Cabe destacar que",
            "Es crucial",
            "Se debe considerar",
            "Es recomendable",
            "Por otro lado",
            "Además de esto",
            "En este sentido",
            "De esta manera",
            "Por lo tanto",
            "En consecuencia",
            "A pesar de",
            "especialmente considerando",
            "utilizando métodos como",
            "Se recomienda",
            "la cantidad promedio es muy baja",
            "se recomienda un sistema de",
            "para una mayor eficiencia en el uso del agua",
            "Las predicciones meteorológicas para las próximas",
            "en la ubicación especificada"
        ]
        
        for phrase in redundant_phrases:
            text = text.replace(phrase + " ", "").replace(phrase + ", ", "").replace(phrase, "")
        
        # Simplificar conectores y expresiones comunes
        replacements = {
            "especialmente considerando": "por",
            "utilizando métodos como": "usando",
            "Se recomienda": "Usar",
            "Es crucial": "Importante:",
            "A pesar de": "Aunque",
            "la cantidad promedio es muy baja": "precipitación baja",
            "se recomienda un sistema de": "usar",
            "para una mayor eficiencia": "eficiente",
            "Es recomendable": "Usar",
            "Se debe considerar": "",
            "la implementación de": "",
            "la medición con": "",
            "la observación visual": "observar",
            "para determinar": "para",
            "las necesidades hídricas": "necesidades de agua",
            "aproximadamente 30 días": "30 días",
            "sugiere condiciones": "indica",
            "aumenta el riesgo de": "favorece",
            "lo que sugiere": "requiere",
            "En general": "",
            "poco propicio para": "inadecuado para"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Limpiar espacios extra y puntuación
        text = " ".join(text.split())  # Normalizar espacios
        text = text.replace("  ", " ").replace(" .", ".").replace(" ,", ",")
        
        return text.strip()
    
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
