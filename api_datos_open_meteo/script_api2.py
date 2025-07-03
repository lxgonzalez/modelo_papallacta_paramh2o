import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import time
import numpy as np
from collections import defaultdict

# Configuración de la sesión con caché y reintentos
cache_session = requests_cache.CachedSession('.cache', expire_after=86400)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Cargar el archivo de precipitaciones limpio
ruta_archivo = 'precipitacion_limpia.csv'
df_precipitacion = pd.read_csv(ruta_archivo)
df_precipitacion['Fecha'] = pd.to_datetime(df_precipitacion['Fecha'])

# Coordenadas de las estaciones
estaciones = {
    'M5023': {'lat': -0.3798, 'lon': -78.1959},
    'M5025': {'lat': -0.3337, 'lon': -78.1985},
    'P34': {'lat': -0.3809, 'lon': -78.1411},
    'P63': {'lat': -0.3206, 'lon': -78.1917}
}

def obtener_datos_por_lotes(fechas_unicas, lat, lon, lote_dias=30):
    """Obtiene datos meteorológicos en lotes grandes para ser MUY rápido."""
    
    print(f"  📦 Obteniendo datos en lotes de {lote_dias} días...")
    
    # Agrupar fechas por rangos
    fechas_ordenadas = sorted(fechas_unicas)
    fecha_inicio = fechas_ordenadas[0].date()
    fecha_fin = fechas_ordenadas[-1].date()
    
    todos_los_datos = {}
    
    # Procesar en lotes de X días
    fecha_actual = fecha_inicio
    lote_num = 1
    
    while fecha_actual <= fecha_fin:
        fecha_fin_lote = min(fecha_actual + timedelta(days=lote_dias-1), fecha_fin)
        
        print(f"    🔄 Lote {lote_num}: {fecha_actual} a {fecha_fin_lote}")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": fecha_actual.strftime("%Y-%m-%d"),
            "end_date": fecha_fin_lote.strftime("%Y-%m-%d"),
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation_sum", "soil_moisture_0_to_10cm"],
            "temperature_unit": "celsius",
            "timezone": "auto"
        }
        
        try:
            time.sleep(0.2)  # Pausa mínima
            responses = openmeteo.weather_api(url, params=params)
            
            if responses:
                response = responses[0]
                hourly = response.Hourly()
                
                # Obtener arrays de datos
                hourly_time = hourly.Time()
                temp_data = hourly.Variables(0).ValuesAsNumpy()
                humidity_data = hourly.Variables(1).ValuesAsNumpy()
                precip_data = hourly.Variables(2).ValuesAsNumpy()  # Datos de precipitación
                soil_data = hourly.Variables(3).ValuesAsNumpy()
                
                # Crear timestamps
                timestamps = pd.date_range(
                    start=pd.to_datetime(hourly_time, unit="s"),
                    periods=len(temp_data),
                    freq="h"
                )
                
                # Almacenar en diccionario para acceso rápido
                for i, timestamp in enumerate(timestamps):
                    todos_los_datos[timestamp] = {
                        'temperatura': temp_data[i] if not np.isnan(temp_data[i]) else np.nan,
                        'humedad': humidity_data[i] if not np.isnan(humidity_data[i]) else np.nan,
                        'precipitacion': precip_data[i] if not np.isnan(precip_data[i]) else np.nan,  # Almacenar precipitación
                        'suelo': soil_data[i] if not np.isnan(soil_data[i]) else np.nan
                    }
                
                print(f"    ✅ Lote {lote_num} completado ({len(timestamps)} horas)")
            
        except Exception as e:
            print(f"    ❌ Error en lote {lote_num}: {str(e)}")
        
        fecha_actual = fecha_fin_lote + timedelta(days=1)
        lote_num += 1
    
    return todos_los_datos

def procesar_estacion_optimizada(nombre_estacion, coords):
    """Procesa una estación de manera ultra-optimizada."""
    
    print(f"\n🚀 PROCESANDO ESTACIÓN {nombre_estacion} (MODO RÁPIDO)")
    print("="*50)
    
    # Obtener fechas únicas para minimizar llamadas a API
    fechas_unicas = df_precipitacion['Fecha'].dt.floor('h').unique()
    print(f"📅 Fechas únicas a procesar: {len(fechas_unicas)}")
    
    # Obtener TODOS los datos de una vez en lotes grandes
    datos_meteorologicos = obtener_datos_por_lotes(fechas_unicas, coords['lat'], coords['lon'])
    
    # Crear DataFrame resultado
    df_resultado = df_precipitacion.copy()
    
    # Inicializar columnas
    df_resultado[f'Temperatura_{nombre_estacion} (°C)'] = np.nan
    df_resultado[f'Humedad_Relativa_{nombre_estacion} (%)'] = np.nan  
    df_resultado[f'Humedad_Suelo_{nombre_estacion} (m³/m³)'] = np.nan
    df_resultado[f'Precipitacion_{nombre_estacion} (mm)'] = np.nan  # Nueva columna para precipitación
    
    # Mapear datos de manera vectorizada (SÚPER RÁPIDO)
    print("  🔄 Mapeando datos...")
    
    for idx, row in df_resultado.iterrows():
        fecha_exacta = row['Fecha']
        
        if fecha_exacta in datos_meteorologicos:
            datos = datos_meteorologicos[fecha_exacta]
            df_resultado.at[idx, f'Temperatura_{nombre_estacion} (°C)'] = datos['temperatura']
            df_resultado.at[idx, f'Humedad_Relativa_{nombre_estacion} (%)'] = datos['humedad']
            df_resultado.at[idx, f'Humedad_Suelo_{nombre_estacion} (m³/m³)'] = datos['suelo']
            df_resultado.at[idx, f'Precipitacion_{nombre_estacion} (mm)'] = datos['precipitacion']  # Asignar precipitación
        
        # Mostrar progreso cada 10000 filas
        if (idx + 1) % 10000 == 0:
            print(f"    📊 Procesadas {idx + 1}/{len(df_resultado)} filas")
    
    return df_resultado

def procesar_todas_estaciones_paralelo():
    """Procesa todas las estaciones de manera optimizada."""
    
    print("🌟 INICIANDO PROCESAMIENTO ULTRA-RÁPIDO")
    print("="*60)
    print(f"📁 Archivo: {ruta_archivo}")
    print(f"📊 Registros totales: {len(df_precipitacion):,}")
    print(f"🗓️  Rango: {df_precipitacion['Fecha'].min()} - {df_precipitacion['Fecha'].max()}")
    
    resultados = {}
    
    # Procesar cada estación
    for nombre_estacion, coords in estaciones.items():
        inicio_estacion = time.time()
        
        df_estacion = procesar_estacion_optimizada(nombre_estacion, coords)
        resultados[nombre_estacion] = df_estacion
        
        # Guardar resultado individual
        archivo_salida = f'precipitacion_meteorologica_{nombre_estacion}.csv'
        df_estacion.to_csv(archivo_salida, index=False)
        
        tiempo_estacion = time.time() - inicio_estacion
        print(f"✅ {nombre_estacion} completada en {tiempo_estacion:.1f} segundos")
        print(f"💾 Guardado: {archivo_salida}")
    
    # Crear archivo consolidado
    print("\n🔄 Creando archivo consolidado...")
    df_consolidado = df_precipitacion.copy()
    
    for nombre_estacion in estaciones.keys():
        df_estacion = resultados[nombre_estacion]
        # Copiar columnas meteorológicas
        for col in df_estacion.columns:
            if col.startswith(('Temperatura_', 'Humedad_Relativa_', 'Humedad_Suelo_', 'Precipitacion_')):
                df_consolidado[col] = df_estacion[col]
    
    # Guardar consolidado
    df_consolidado.to_csv('precipitacion_meteorologica_completa.csv', index=False)
    print("💾 Archivo consolidado: precipitacion_meteorologica_completa.csv")
    
    return df_consolidado

def mostrar_resumen_optimizado(df):
    """Muestra resumen de datos obtenidos."""
    print("\n📊 RESUMEN FINAL")
    print("="*40)
    
    for estacion in estaciones.keys():
        temp_col = f'Temperatura_{estacion} (°C)'
        hum_col = f'Humedad_Relativa_{estacion} (%)'
        soil_col = f'Humedad_Suelo_{estacion} (m³/m³)'
        precip_col = f'Precipitacion_{estacion} (mm)'  # Nueva columna para precipitación
        
        if temp_col in df.columns:
            temp_ok = df[temp_col].notna().sum()
            hum_ok = df[hum_col].notna().sum()
            soil_ok = df[soil_col].notna().sum()
            precip_ok = df[precip_col].notna().sum()  # Contar precipitación
            total = len(df)
            
            print(f"\n📍 {estacion}:")
            print(f"  🌡️  Temperatura: {temp_ok:,}/{total:,} ({temp_ok/total*100:.1f}%)")
            print(f"  💧 Humedad Rel.: {hum_ok:,}/{total:,} ({hum_ok/total*100:.1f}%)")
            print(f"  🌱 Humedad Suelo: {soil_ok:,}/{total:,} ({soil_ok/total*100:.1f}%)")
            print(f"  🌧️ Precipitación: {precip_ok:,}/{total:,} ({precip_ok/total*100:.1f}%)")  # Resumen de precipitación

# EJECUTAR PROCESAMIENTO OPTIMIZADO
if __name__ == "__main__":
    inicio_total = time.time()
    
    # Procesar todas las estaciones
    df_final = procesar_todas_estaciones_paralelo()
    
    # Mostrar resumen
    mostrar_resumen_optimizado(df_final)
    
    tiempo_total = time.time() - inicio_total
    print(f"\n🎉 ¡COMPLETADO EN {tiempo_total:.1f} SEGUNDOS!")
    print(f"⚡ Velocidad: {len(df_precipitacion)/tiempo_total:.0f} registros/segundo")
