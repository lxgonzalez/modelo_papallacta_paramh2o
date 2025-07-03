import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Cargar los archivos de datos con los valores rellenos
df_M5023 = pd.read_csv('datos_M5023_filled.csv')
df_M5025 = pd.read_csv('datos_M5025_filled.csv')
df_P34 = pd.read_csv('datos_P34_filled.csv')
df_P63 = pd.read_csv('datos_P63_filled.csv')

# Convertir la columna de 'Fecha' a datetime para alinear por fecha
df_M5023['Fecha'] = pd.to_datetime(df_M5023['Fecha'])
df_M5025['Fecha'] = pd.to_datetime(df_M5025['Fecha'])
df_P34['Fecha'] = pd.to_datetime(df_P34['Fecha'])
df_P63['Fecha'] = pd.to_datetime(df_P63['Fecha'])

# Función para entrenar y evaluar el modelo para predecir cada variable
def train_and_predict(df, station_name):
    # Entrenamos tres modelos para predecir Temperatura, Humedad y Precipitación

    # **Modelo para Precipitación (mm)**
    X_precip = df[['Temperatura (°C)', 'Humedad_Relativa (%)']]  # Variables predictoras
    y_precip = df['Precipitacion (mm)']  # Variable a predecir (Precipitación)
    
    # Entrenar el modelo de Precipitación
    model_precip = RandomForestRegressor(n_estimators=100, random_state=42)
    model_precip.fit(X_precip, y_precip)
    y_precip_pred = model_precip.predict(X_precip)

    # **Modelo para Temperatura (°C)**
    X_temp = df[['Humedad_Relativa (%)', 'Precipitacion (mm)']]  # Variables predictoras
    y_temp = df['Temperatura (°C)']  # Variable a predecir (Temperatura)
    
    # Entrenar el modelo de Temperatura
    model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_temp.fit(X_temp, y_temp)
    y_temp_pred = model_temp.predict(X_temp)

    # **Modelo para Humedad Relativa (%)**
    X_humidity = df[['Temperatura (°C)', 'Precipitacion (mm)']]  # Variables predictoras
    y_humidity = df['Humedad_Relativa (%)']  # Variable a predecir (Humedad Relativa)
    
    # Entrenar el modelo de Humedad Relativa
    model_humidity = RandomForestRegressor(n_estimators=100, random_state=42)
    model_humidity.fit(X_humidity, y_humidity)
    y_humidity_pred = model_humidity.predict(X_humidity)

    # Evaluación de los modelos (para cada uno)
    mae_precip = mean_absolute_error(y_precip, y_precip_pred)
    rmse_precip = np.sqrt(mean_squared_error(y_precip, y_precip_pred))

    mae_temp = mean_absolute_error(y_temp, y_temp_pred)
    rmse_temp = np.sqrt(mean_squared_error(y_temp, y_temp_pred))

    mae_humidity = mean_absolute_error(y_humidity, y_humidity_pred)
    rmse_humidity = np.sqrt(mean_squared_error(y_humidity, y_humidity_pred))

    # Mostrar los resultados
    print(f"Resultados del modelo para la estación {station_name}:")
    print(f"**Precipitación**:")
    print(f"MAE: {mae_precip}, RMSE: {rmse_precip}")
    print(f"**Temperatura**:")
    print(f"MAE: {mae_temp}, RMSE: {rmse_temp}")
    print(f"**Humedad Relativa**:")
    print(f"MAE: {mae_humidity}, RMSE: {rmse_humidity}")

    # Guardar los modelos entrenados
    joblib.dump(model_precip, f'{station_name}_modelo_precip.pkl')
    joblib.dump(model_temp, f'{station_name}_modelo_temp.pkl')
    joblib.dump(model_humidity, f'{station_name}_modelo_humidity.pkl')
    print(f"Modelos guardados como {station_name}_modelo_precip.pkl, {station_name}_modelo_temp.pkl, {station_name}_modelo_humidity.pkl")

    # Gráficos de Predicción vs Real para cada modelo
    plt.figure(figsize=(10, 6))
    plt.scatter(y_precip, y_precip_pred, color='blue', label='Predicción de Precipitación')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicción de Precipitación vs Reales - {station_name}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_temp, y_temp_pred, color='green', label='Predicción de Temperatura')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicción de Temperatura vs Reales - {station_name}')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_humidity, y_humidity_pred, color='red', label='Predicción de Humedad')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title(f'Predicción de Humedad vs Reales - {station_name}')
    plt.legend()
    plt.show()

# Entrenar y predecir para cada estación
train_and_predict(df_M5023, 'M5023')
train_and_predict(df_M5025, 'M5025')
train_and_predict(df_P34, 'P34')
train_and_predict(df_P63, 'P63')
