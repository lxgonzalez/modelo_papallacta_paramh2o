# API de Predicciones Meteorológicas y Análisis Agrícola

Esta API combina modelos de predicción meteorológica LSTM con análisis agrícola inteligente usando Google Gemini. Proporciona predicciones meteorológicas precisas y recomendaciones agrícolas personalizadas para cualquier ubicación geográfica.

## 🌟 Características Principales

- **Predicciones Meteorológicas**: Usa modelos LSTM entrenados en datos de 4 estaciones meteorológicas
- **Selección Automática de Modelo**: Selecciona automáticamente el modelo de la estación más cercana
- **Análisis Agrícola con IA**: Proporciona recomendaciones agrícolas usando Google Gemini
- **Datos Históricos**: Consulta datos meteorológicos históricos de Open-Meteo
- **API RESTful**: Interfaz simple y bien documentada
- **Arquitectura Clean Code**: Código modular, escalable y mantenible

## 🏗️ Arquitectura del Proyecto

La aplicación sigue principios de Clean Code con una arquitectura modular:

```
src/
├── config/          # Configuraciones y constantes
│   └── settings.py
├── models/          # Modelos de datos y validaciones
│   └── data_models.py
├── services/        # Lógica de negocio
│   ├── station_service.py
│   ├── weather_service.py
│   ├── prediction_service.py
│   └── analysis_service.py
├── routes/          # Endpoints de la API
│   ├── weather_routes.py
│   ├── prediction_routes.py
│   ├── station_routes.py
│   └── info_routes.py
├── utils/           # Utilidades y helpers
│   ├── validators.py
│   └── logging.py
└── service_manager.py  # Gestor de servicios

app.py              # Aplicación principal
```

### Principios Aplicados

- **Single Responsibility**: Cada clase tiene una responsabilidad específica
- **Dependency Injection**: Los servicios se inyectan donde se necesitan
- **Separation of Concerns**: Separación clara entre rutas, servicios y configuración
- **Error Handling**: Manejo centralizado de errores y logging
- **Type Hints**: Tipado para mejor mantenibilidad
- **Dataclasses**: Modelos de datos inmutables y validados

## 📋 Requisitos

### Dependencias Python
```bash
pip install -r requirements.txt
```

### Variables de Entorno
```bash
# Configurar API key de Google Gemini (opcional pero recomendado)
export GEMINI_API_KEY="tu_api_key_aqui"

# En Windows PowerShell:
$env:GEMINI_API_KEY="tu_api_key_aqui"
```

## 🚀 Instalación y Ejecución

1. **Clonar el repositorio y navegar al directorio**:
```bash
cd modelo_papallacta_paramh2o
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Configurar API key de Gemini** (opcional):
Crear archivo `.env` en la raíz del proyecto:
```
GEMINI_API_KEY=tu_api_key_aqui
```

4. **Ejecutar el servidor**:
```bash
python app.py
```

El servidor estará disponible en `http://localhost:5000`

## 🔌 Endpoints de la API

### 1. Información General
**GET** `/`
- **Descripción**: Información sobre la API y sus capacidades
- **Respuesta**: Detalles de endpoints, estaciones disponibles y formato de peticiones

### 2. Estado del Servidor
**GET** `/health`
- **Descripción**: Verificar que el servidor esté funcionando correctamente
- **Respuesta**: Estado del servidor, servicios y configuración

### 3. Información de Estaciones
**GET** `/stations`
- **Descripción**: Información sobre todas las estaciones meteorológicas disponibles
- **Respuesta**: Lista de estaciones disponibles y no disponibles con sus coordenadas

### 4. Opciones de Análisis Agrícola
**GET** `/analysis-options`
- **Descripción**: Obtener todas las opciones de análisis agrícola disponibles
- **Respuesta**: Lista de tipos de análisis con descripciones

**Tipos de análisis disponibles:**
- `general`: Condiciones generales del clima
- `cultivos`: Recomendaciones de cultivos apropiados
- `riego`: Manejo y programación del riego
- `alertas`: Alertas climáticas y riesgos
- `cronograma`: Cronograma agrícola óptimo
- `plagas`: Manejo de plagas según condiciones climáticas
- `suelo`: Conservación y manejo del suelo

### 5. Datos Meteorológicos Históricos
**POST** `/weather_data`
- **Descripción**: Obtener datos meteorológicos históricos (últimos 30 días)
- **Parámetros**:
  ```json
  {
    "date": "2024-06-01",
    "latitude": -0.35,
    "longitude": -78.17
  }
  ```
- **Respuesta**: Datos históricos de temperatura, precipitación y humedad

### 6. Predicciones con Análisis Agrícola (Endpoint Principal)
**POST** `/predict`
- **Descripción**: Realizar predicciones meteorológicas y análisis agrícola completo
- **Parámetros**:
  ```json
  {
    "date": "2024-06-01",
    "latitude": -0.35,
    "longitude": -78.17,
    "include_analysis": true,
    "analysis_types": ["general", "cultivos", "riego", "alertas"]
  }
  ```

**Parámetros detallados:**
- `date` (requerido): Fecha de referencia en formato YYYY-MM-DD
- `latitude` (requerido): Latitud en decimal (-90 a 90)
- `longitude` (requerido): Longitud en decimal (-180 a 180)
- `include_analysis` (opcional): Incluir análisis agrícola (default: true)
- `analysis_types` (opcional): Lista de tipos de análisis específicos

### 7. Encontrar Estación Más Cercana
**POST** `/nearest_station`
- **Descripción**: Encontrar la estación meteorológica más cercana a unas coordenadas
- **Parámetros**:
  ```json
  {
    "latitude": -0.35,
    "longitude": -78.17
  }
  ```
- **Respuesta**: Estación más cercana y distancias a todas las estaciones

## 🎯 Ejemplos de Uso

### Predicción Completa con Análisis
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-06-01",
    "latitude": -0.35,
    "longitude": -78.17,
    "include_analysis": true,
    "analysis_types": ["general", "cultivos", "riego"]
  }'
```

### Solo Predicciones (sin análisis)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-06-01",
    "latitude": -0.35,
    "longitude": -78.17,
    "include_analysis": false
  }'
```

### Consultar Datos Históricos
```bash
curl -X POST http://localhost:5000/weather_data \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-06-01",
    "latitude": -0.35,
    "longitude": -78.17
  }'
```

## 🗺️ Estaciones Meteorológicas

La API utiliza modelos entrenados en 4 estaciones meteorológicas:

| Estación | Latitud  | Longitud | Descripción |
|----------|----------|----------|-------------|
| M5023    | -0.3798  | -78.1959 | Estación Meteorológica 1 |
| M5025    | -0.3337  | -78.1985 | Estación Meteorológica 2 |
| P34      | -0.3809  | -78.1411 | Estación Pluviométrica 1 |
| P63      | -0.3206  | -78.1917 | Estación Pluviométrica 2 |

## 🧠 Análisis Agrícola con IA

El análisis agrícola utiliza Google Gemini para proporcionar:

1. **Evaluación Climática**: Análisis de las condiciones meteorológicas previstas
2. **Recomendaciones de Cultivos**: Qué cultivos son más apropiados para las condiciones
3. **Manejo del Riego**: Estrategias de riego basadas en precipitación prevista
4. **Alertas Climáticas**: Identificación de riesgos como sequías o exceso de humedad
5. **Cronograma Agrícola**: Momentos óptimos para siembra, cosecha y otras actividades
6. **Manejo de Plagas**: Condiciones que podrían favorecer el desarrollo de plagas
7. **Conservación del Suelo**: Medidas preventivas según las condiciones climáticas

## 🔧 Desarrollo y Extensión

### Agregar Nuevos Servicios

1. Crear el servicio en `src/services/`
2. Agregarlo al `ServiceManager`
3. Crear las rutas correspondientes
4. Registrar las rutas en `app.py`

### Agregar Nuevas Validaciones

1. Agregar validadores en `src/utils/validators.py`
2. Usar en los modelos de datos
3. Aplicar en las rutas

### Configuración

Toda la configuración está centralizada en `src/config/settings.py`:
- Configuración de la aplicación
- Configuración de estaciones
- Configuración de análisis
- URLs y timeouts de APIs externas

## ⚠️ Manejo de Errores

La API implementa manejo robusto de errores:

- **Validación de Entrada**: Validación completa de todos los parámetros
- **Logging Centralizado**: Todos los errores se registran con contexto
- **Respuestas Consistentes**: Formato estándar para todas las respuestas
- **Códigos de Estado HTTP**: Uso apropiado de códigos de estado

## 🚨 Limitaciones y Consideraciones

1. **Cobertura Geográfica**: Optimizada para la región de Papallacta, Ecuador
2. **Horizonte de Predicción**: 715 horas (aproximadamente 30 días)
3. **Dependencia de APIs Externas**: Usa Open-Meteo para datos históricos
4. **Análisis IA**: Requiere configuración de Gemini API key
5. **Modelos Preentrenados**: Los modelos LSTM están preentrenados y no se actualizan automáticamente

## 🔍 Monitoreo y Logging

El sistema incluye logging comprensivo:
- Logs de performance con métricas de tiempo
- Logs de errores con contexto completo
- Logs de información para debugging
- Estructura de logs consistente

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Hacer fork del repositorio
2. Crear una rama para tu feature
3. Seguir las convenciones de Clean Code
4. Agregar tests para nuevas funcionalidades
5. Hacer commit de tus cambios
6. Hacer push a la rama
7. Crear un Pull Request

### Convenciones de Código

- Usar type hints en todas las funciones
- Documentar todas las clases y métodos
- Seguir PEP 8 para estilo de código
- Usar dataclasses para modelos de datos
- Implementar manejo de errores apropiado

## 📜 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

## 📞 Soporte

Para soporte técnico o preguntas:
- Crear un issue en el repositorio
- Revisar la documentación de endpoints
- Verificar los logs del servidor para errores específicos

---

**Nota**: Esta API está diseñada con principios de Clean Code para facilitar el mantenimiento, testing y extensión. La arquitectura modular permite agregar nuevas funcionalidades fácilmente.
