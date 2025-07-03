import pandas as pd
import numpy as np

def crear_csvs_individuales_por_estacion():
    """
    Crea un CSV limpio para cada estación con:
    - Fecha
    - Precipitación de esa estación
    - Temperatura de esa estación  
    - Humedad Relativa de esa estación
    - Humedad del Suelo de esa estación
    """
    
    print("🔄 Reorganizando datos por estación individual...")
    
    # Lista de estaciones
    estaciones = ['M5023', 'M5025', 'P34', 'P63']
    
    for estacion in estaciones:
        print(f"\n📊 Procesando estación: {estacion}")
        
        # Leer el archivo específico de la estación
        archivo_entrada = f'precipitacion_meteorologica_{estacion}.csv'
        
        try:
            df = pd.read_csv(archivo_entrada)
            print(f"  ✅ Archivo leído: {len(df)} registros")
            
            # Crear DataFrame limpio para esta estación
            df_estacion = pd.DataFrame()
            
            # Columna de fecha
            df_estacion['Fecha'] = df['Fecha']
            
            # Columna de precipitación específica de esta estación
            if estacion in df.columns:
                df_estacion['Precipitacion (mm)'] = df[estacion]
            else:
                print(f"  ⚠️  Columna {estacion} no encontrada, llenando con NaN")
                df_estacion['Precipitacion (mm)'] = np.nan
            
            # Columnas meteorológicas específicas de esta estación
            temp_col = f'Temperatura_{estacion} (°C)'
            hum_col = f'Humedad_Relativa_{estacion} (%)'
            soil_col = f'Humedad_Suelo_{estacion} (m³/m³)'
            
            # Agregar datos meteorológicos
            if temp_col in df.columns:
                df_estacion['Temperatura (°C)'] = df[temp_col]
            else:
                df_estacion['Temperatura (°C)'] = np.nan
                print(f"  ⚠️  {temp_col} no encontrada")
            
            if hum_col in df.columns:
                df_estacion['Humedad_Relativa (%)'] = df[hum_col]
            else:
                df_estacion['Humedad_Relativa (%)'] = np.nan
                print(f"  ⚠️  {hum_col} no encontrada")
            
            if soil_col in df.columns:
                df_estacion['Humedad_Suelo (m³/m³)'] = df[soil_col]
            else:
                df_estacion['Humedad_Suelo (m³/m³)'] = np.nan
                print(f"  ⚠️  {soil_col} no encontrada")
            
            # Guardar archivo limpio
            archivo_salida = f'datos_finales_{estacion}.csv'
            df_estacion.to_csv(archivo_salida, index=False)
            
            # Mostrar estadísticas
            precipitacion_datos = df_estacion['Precipitacion (mm)'].notna().sum()
            temperatura_datos = df_estacion['Temperatura (°C)'].notna().sum()
            humedad_datos = df_estacion['Humedad_Relativa (%)'].notna().sum()
            suelo_datos = df_estacion['Humedad_Suelo (m³/m³)'].notna().sum()
            total = len(df_estacion)
            
            print(f"  📈 Estadísticas de {estacion}:")
            print(f"    • Precipitación: {precipitacion_datos:,}/{total:,} ({precipitacion_datos/total*100:.1f}%)")
            print(f"    • Temperatura: {temperatura_datos:,}/{total:,} ({temperatura_datos/total*100:.1f}%)")
            print(f"    • Humedad Rel.: {humedad_datos:,}/{total:,} ({humedad_datos/total*100:.1f}%)")
            print(f"    • Humedad Suelo: {suelo_datos:,}/{total:,} ({suelo_datos/total*100:.1f}%)")
            print(f"  💾 Guardado: {archivo_salida}")
            
            # Mostrar primeras filas como ejemplo
            print(f"  👁️  Primeras 3 filas de {estacion}:")
            print(df_estacion.head(3).to_string(index=False))
            
        except FileNotFoundError:
            print(f"  ❌ Archivo no encontrado: {archivo_entrada}")
        except Exception as e:
            print(f"  ❌ Error procesando {estacion}: {str(e)}")

def crear_resumen_consolidado():
    """Crea un resumen consolidado con estadísticas de todas las estaciones."""
    
    print("\n📋 Creando resumen consolidado...")
    
    estaciones = ['M5023', 'M5025', 'P34', 'P63']
    resumen_data = []
    
    for estacion in estaciones:
        archivo = f'datos_finales_{estacion}.csv'
        try:
            df = pd.read_csv(archivo)
            
            resumen_data.append({
                'Estacion': estacion,
                'Total_Registros': len(df),
                'Precipitacion_Disponible': df['Precipitacion (mm)'].notna().sum(),
                'Temperatura_Disponible': df['Temperatura (°C)'].notna().sum(),
                'Humedad_Rel_Disponible': df['Humedad_Relativa (%)'].notna().sum(),
                'Humedad_Suelo_Disponible': df['Humedad_Suelo (m³/m³)'].notna().sum(),
                'Fecha_Inicio': df['Fecha'].iloc[0],
                'Fecha_Fin': df['Fecha'].iloc[-1],
                'Precipitacion_Promedio': df['Precipitacion (mm)'].mean(),
                'Temperatura_Promedio': df['Temperatura (°C)'].mean(),
                'Humedad_Rel_Promedio': df['Humedad_Relativa (%)'].mean(),
                'Humedad_Suelo_Promedio': df['Humedad_Suelo (m³/m³)'].mean()
            })
            
        except Exception as e:
            print(f"  ⚠️  Error leyendo {archivo}: {str(e)}")
    
    if resumen_data:
        df_resumen = pd.DataFrame(resumen_data)
        df_resumen.to_csv('resumen_estaciones.csv', index=False)
        print("  💾 Resumen guardado: resumen_estaciones.csv")
        
        print("\n📊 RESUMEN FINAL:")
        print("="*70)
        for _, row in df_resumen.iterrows():
            print(f"\n🏭 ESTACIÓN {row['Estacion']}:")
            print(f"  📅 Período: {row['Fecha_Inicio']} - {row['Fecha_Fin']}")
            print(f"  📊 Registros: {row['Total_Registros']:,}")
            print(f"  🌧️  Precipitación: {row['Precipitacion_Disponible']:,} datos (promedio: {row['Precipitacion_Promedio']:.2f} mm)")
            print(f"  🌡️  Temperatura: {row['Temperatura_Disponible']:,} datos (promedio: {row['Temperatura_Promedio']:.1f}°C)")
            print(f"  💧 Humedad Rel.: {row['Humedad_Rel_Disponible']:,} datos (promedio: {row['Humedad_Rel_Promedio']:.1f}%)")
            print(f"  🌱 Humedad Suelo: {row['Humedad_Suelo_Disponible']:,} datos (promedio: {row['Humedad_Suelo_Promedio']:.3f} m³/m³)")

def verificar_estructura_archivos():
    """Verifica qué archivos están disponibles y su estructura."""
    
    print("🔍 Verificando archivos disponibles...")
    
    import os
    archivos_csv = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    print(f"📁 Archivos CSV encontrados: {len(archivos_csv)}")
    for archivo in sorted(archivos_csv):
        print(f"  • {archivo}")
    
    # Verificar estructura de archivos meteorológicos
    estaciones = ['M5023', 'M5025', 'P34', 'P63']
    
    print(f"\n🔍 Verificando archivos meteorológicos...")
    for estacion in estaciones:
        archivo = f'precipitacion_meteorologica_{estacion}.csv'
        if archivo in archivos_csv:
            try:
                df = pd.read_csv(archivo, nrows=0)  # Solo leer headers
                print(f"  ✅ {archivo}: {len(df.columns)} columnas")
                print(f"    Columnas: {list(df.columns)}")
            except Exception as e:
                print(f"  ❌ Error leyendo {archivo}: {str(e)}")
        else:
            print(f"  ❌ No encontrado: {archivo}")

# EJECUTAR REORGANIZACIÓN
if __name__ == "__main__":
    print("🚀 REORGANIZANDO DATOS POR ESTACIÓN INDIVIDUAL")
    print("="*60)
    
    # Verificar archivos disponibles
    verificar_estructura_archivos()
    
    # Crear CSVs individuales por estación
    crear_csvs_individuales_por_estacion()
    
    # Crear resumen consolidado
    crear_resumen_consolidado()
    
    print(f"\n🎉 ¡REORGANIZACIÓN COMPLETADA!")
    print("📁 Archivos generados:")
    print("  • datos_finales_M5023.csv")
    print("  • datos_finales_M5025.csv") 
    print("  • datos_finales_P34.csv")
    print("  • datos_finales_P63.csv")
    print("  • resumen_estaciones.csv")