import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Cargar los archivos de datos finales
df_M5023 = pd.read_csv('datos_finales_M5023.csv')
df_M5025 = pd.read_csv('datos_finales_M5025.csv')
df_P34 = pd.read_csv('datos_finales_P34.csv')
df_P63 = pd.read_csv('datos_finales_P63.csv')

# Convertir la columna de 'Fecha' a datetime para alinear por fecha
df_M5023['Fecha'] = pd.to_datetime(df_M5023['Fecha'])
df_M5025['Fecha'] = pd.to_datetime(df_M5025['Fecha'])
df_P34['Fecha'] = pd.to_datetime(df_P34['Fecha'])
df_P63['Fecha'] = pd.to_datetime(df_P63['Fecha'])

# Función para rellenar valores faltantes usando Random Forest
def fill_missing_values(df):
    # Creamos una copia de los datos para no modificar el original
    df_filled = df.copy()

    # Detectar filas con NaN en 'Precipitacion (mm)' (objetivo)
    missing_data = df_filled[df_filled['Precipitacion (mm)'].isnull()]

    # Entrenamiento solo con los datos no faltantes
    df_train = df_filled.dropna(subset=['Precipitacion (mm)'])

    # Definir las variables predictoras (Temperatura, Humedad Relativa)
    X_train = df_train[['Temperatura (°C)', 'Humedad_Relativa (%)']]  # Variables predictoras
    y_train = df_train['Precipitacion (mm)']  # Variable a predecir

    # Entrenamiento del modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predecir los valores faltantes
    X_missing = missing_data[['Temperatura (°C)', 'Humedad_Relativa (%)']]
    predicted_precipitation = model.predict(X_missing)

    # Reemplazar los NaN con las predicciones
    df_filled.loc[df_filled['Precipitacion (mm)'].isnull(), 'Precipitacion (mm)'] = predicted_precipitation

    return df_filled

# Rellenar los valores faltantes para cada estación
df_M5023_filled = fill_missing_values(df_M5023)
df_M5025_filled = fill_missing_values(df_M5025)
df_P34_filled = fill_missing_values(df_P34)
df_P63_filled = fill_missing_values(df_P63)

# Guardar los resultados rellenos en nuevos archivos CSV
df_M5023_filled.to_csv('datos_M5023_filled.csv', index=False)
df_M5025_filled.to_csv('datos_M5025_filled.csv', index=False)
df_P34_filled.to_csv('datos_P34_filled.csv', index=False)
df_P63_filled.to_csv('datos_P63_filled.csv', index=False)

# Visualización de los resultados de la interpolación vs predicción
def plot_and_save(df_original, df_filled, station_name):
    """Función para graficar antes vs después de la predicción"""
    plt.figure(figsize=(14, 6))
    plt.plot(df_original['Fecha'], df_original['Precipitacion (mm)'], label='Original', marker='o', linestyle='-', color='blue')
    plt.plot(df_filled['Fecha'], df_filled['Precipitacion (mm)'], label='Predicción', marker='x', linestyle='-', color='green')
    
    # Agregar título y etiquetas
    plt.title(f'Precipitación {station_name} - Original vs Predicción (RF)')
    plt.xlabel('Fecha')
    plt.ylabel('Precipitación (mm)')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Guardar la gráfica como imagen
    file_name = f'precipitacion_{station_name}_prediccion.png'
    plt.savefig(file_name)
    plt.show()
    print(f"Gráfico guardado en {file_name}")

# Graficar y guardar los resultados para cada estación
plot_and_save(df_M5023, df_M5023_filled, 'M5023')
plot_and_save(df_M5025, df_M5025_filled, 'M5025')
plot_and_save(df_P34, df_P34_filled, 'P34')
plot_and_save(df_P63, df_P63_filled, 'P63')

print("✅ Archivos CSV generados con los datos rellenos")
