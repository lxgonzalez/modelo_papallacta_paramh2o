import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib  # Para guardar y cargar el scaler
import os

# Función para guardar el scaler
def save_scaler(scaler, station_name):
    joblib.dump(scaler, f'{station_name}_scaler.pkl')

# Función para cargar el scaler
def load_scaler(station_name):
    return joblib.load(f'{station_name}_scaler.pkl')

# Cargar los datos de cada estación
df_M5023 = pd.read_csv('datos_M5023_filled.csv')
df_M5025 = pd.read_csv('datos_M5025_filled.csv')
df_P34 = pd.read_csv('datos_P34_filled.csv')
df_P63 = pd.read_csv('datos_P63_filled.csv')

# Convertir la columna de 'Fecha' a datetime
df_M5023['Fecha'] = pd.to_datetime(df_M5023['Fecha'])
df_M5025['Fecha'] = pd.to_datetime(df_M5025['Fecha'])
df_P34['Fecha'] = pd.to_datetime(df_P34['Fecha'])
df_P63['Fecha'] = pd.to_datetime(df_P63['Fecha'])

# Función para preprocesar los datos y normalizarlos
def preprocess_data(df, feature_cols, station_name):
    # Normalizar las características
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[feature_cols].values)
    
    # Guardar el scaler para usarlo después
    save_scaler(scaler, station_name)
    
    # Crear las secuencias para el modelo LSTM
    def create_sequences(data, time_step=30):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:i + time_step])
            y.append(data[i + time_step])  # Todas las columnas como objetivo
        return np.array(X), np.array(y)

    X, y = create_sequences(df_scaled)
    return X, y, scaler

# Crear secuencias para cada estación y normalizar
feature_cols = ['Precipitacion (mm)', 'Temperatura (°C)', 'Humedad_Relativa (%)']
X_M5023, y_M5023, scaler_M5023 = preprocess_data(df_M5023, feature_cols, 'M5023')
X_M5025, y_M5025, scaler_M5025 = preprocess_data(df_M5025, feature_cols, 'M5025')
X_P34, y_P34, scaler_P34 = preprocess_data(df_P34, feature_cols, 'P34')
X_P63, y_P63, scaler_P63 = preprocess_data(df_P63, feature_cols, 'P63')

# Función para construir y entrenar el modelo LSTM multivariado
def create_lstm_model(X_train, y_train, epochs=10, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # CAMBIO IMPORTANTE: Ahora la capa de salida tiene 3 unidades (para las 3 variables)
    model.add(Dense(units=3))  # 3 salidas: precipitación, temperatura, humedad
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Función para guardar modelos LSTM (usar Keras save en lugar de joblib)
def save_lstm_model(model, station_name):
    model.save(f'{station_name}_lstm_model.h5')

# Función para cargar modelos LSTM
def load_lstm_model(station_name):
    from tensorflow.keras.models import load_model
    return load_model(f'{station_name}_lstm_model.h5')

# Función para verificar si es un modelo LSTM
def is_lstm_model(station_name):
    return os.path.exists(f'{station_name}_lstm_model.h5')

# Función para eliminar modelos antiguos (Random Forest)
def remove_old_models():
    old_models = [
        'M5023_modelo_precip.pkl',
        'M5025_modelo_precip.pkl', 
        'P34_modelo_precip.pkl',
        'P63_modelo_precip.pkl'
    ]
    
    for model_file in old_models:
        if os.path.exists(model_file):
            os.remove(model_file)
            print(f"Modelo antiguo {model_file} eliminado.")

# Entrenar y guardar el modelo para cada estación
def train_and_save_lstm_models():
    # Eliminar modelos antiguos primero
    remove_old_models()
    
    print("Entrenando modelo M5023...")
    model_M5023 = create_lstm_model(X_M5023, y_M5023)
    save_lstm_model(model_M5023, 'M5023')
    print("Modelo LSTM M5023 entrenado y guardado.")
    
    print("Entrenando modelo M5025...")
    model_M5025 = create_lstm_model(X_M5025, y_M5025)
    save_lstm_model(model_M5025, 'M5025')
    print("Modelo LSTM M5025 entrenado y guardado.")
    
    print("Entrenando modelo P34...")
    model_P34 = create_lstm_model(X_P34, y_P34)
    save_lstm_model(model_P34, 'P34')
    print("Modelo LSTM P34 entrenado y guardado.")
    
    print("Entrenando modelo P63...")
    model_P63 = create_lstm_model(X_P63, y_P63)
    save_lstm_model(model_P63, 'P63')
    print("Modelo LSTM P63 entrenado y guardado.")

# Cargar o entrenar modelos
try:
    # Verificar si tenemos modelos LSTM guardados
    if all(is_lstm_model(station) for station in ['M5023', 'M5025', 'P34', 'P63']):
        model_M5023 = load_lstm_model('M5023')
        model_M5025 = load_lstm_model('M5025')
        model_P34 = load_lstm_model('P34')
        model_P63 = load_lstm_model('P63')
        print("Modelos LSTM cargados exitosamente.")
    else:
        print("No se encontraron modelos LSTM. Entrenando nuevos modelos.")
        train_and_save_lstm_models()
        # Cargar los modelos recién entrenados
        model_M5023 = load_lstm_model('M5023')
        model_M5025 = load_lstm_model('M5025')
        model_P34 = load_lstm_model('P34')
        model_P63 = load_lstm_model('P63')
        
except Exception as e:
    print(f"Error cargando modelos: {e}")
    print("Entrenando nuevos modelos LSTM.")
    train_and_save_lstm_models()
    # Cargar los modelos recién entrenados
    model_M5023 = load_lstm_model('M5023')
    model_M5025 = load_lstm_model('M5025')
    model_P34 = load_lstm_model('P34')
    model_P63 = load_lstm_model('P63')

# Función para hacer predicciones con el modelo LSTM multivariado
def predict_lstm_multivariate(model, X_test, scaler, y_test=None):
    predictions = model.predict(X_test)
    
    # Desnormalizar las predicciones (ahora tenemos 3 columnas)
    predictions_denorm = scaler.inverse_transform(predictions)
    
    if y_test is not None:
        # Desnormalizar los valores reales
        y_test_denorm = scaler.inverse_transform(y_test)
        
        # Calcular métricas para cada variable
        variables = ['Precipitación', 'Temperatura', 'Humedad']
        metrics = {}
        
        for i, var in enumerate(variables):
            mae = mean_absolute_error(y_test_denorm[:, i], predictions_denorm[:, i])
            rmse = np.sqrt(mean_squared_error(y_test_denorm[:, i], predictions_denorm[:, i]))
            metrics[var] = {'MAE': mae, 'RMSE': rmse}
            print(f"{var} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return predictions_denorm, y_test_denorm, metrics
    
    return predictions_denorm

# Hacer predicciones con el modelo entrenado para cada estación
print("\n" + "="*50)
print("EVALUACIÓN DE MODELOS")
print("="*50)

print("\n--- Predicciones M5023 ---")
predictions_M5023, y_real_M5023, metrics_M5023 = predict_lstm_multivariate(
    model_M5023, X_M5023, scaler_M5023, y_M5023)

print("\n--- Predicciones M5025 ---")
predictions_M5025, y_real_M5025, metrics_M5025 = predict_lstm_multivariate(
    model_M5025, X_M5025, scaler_M5025, y_M5025)

print("\n--- Predicciones P34 ---")
predictions_P34, y_real_P34, metrics_P34 = predict_lstm_multivariate(
    model_P34, X_P34, scaler_P34, y_P34)

print("\n--- Predicciones P63 ---")
predictions_P63, y_real_P63, metrics_P63 = predict_lstm_multivariate(
    model_P63, X_P63, scaler_P63, y_P63)

# Función mejorada para graficar resultados
def plot_predictions_multivariate(y_real, predictions, station_name):
    variables = ['Precipitación (mm)', 'Temperatura (°C)', 'Humedad Relativa (%)']
    colors = ['blue', 'red', 'green']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Predicciones vs Valores Reales - Estación {station_name}', fontsize=16)
    
    for i, (var, color) in enumerate(zip(variables, colors)):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(y_real[:, i], predictions[:, i], color=color, alpha=0.6, s=10)
        
        # Línea diagonal para referencia (predicción perfecta)
        min_val = min(y_real[:, i].min(), predictions[:, i].min())
        max_val = max(y_real[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=1)
        
        ax.set_xlabel(f'Valores Reales - {var}')
        ax.set_ylabel(f'Predicciones - {var}')
        ax.set_title(var)
        ax.grid(True, alpha=0.3)
        
        # Calcular R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_real[:, i], predictions[:, i])
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Función para graficar series temporales
def plot_time_series(y_real, predictions, station_name, max_points=500):
    variables = ['Precipitación (mm)', 'Temperatura (°C)', 'Humedad Relativa (%)']
    colors_real = ['darkblue', 'darkred', 'darkgreen']
    colors_pred = ['lightblue', 'lightcoral', 'lightgreen']
    
    # Limitar puntos para mejor visualización
    if len(y_real) > max_points:
        indices = np.linspace(0, len(y_real)-1, max_points, dtype=int)
        y_real_plot = y_real[indices]
        predictions_plot = predictions[indices]
    else:
        y_real_plot = y_real
        predictions_plot = predictions
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'Series Temporales - Estación {station_name}', fontsize=16)
    
    for i, var in enumerate(variables):
        ax = axes[i]
        x_axis = range(len(y_real_plot))
        
        ax.plot(x_axis, y_real_plot[:, i], color=colors_real[i], 
                label=f'{var} Real', linewidth=1, alpha=0.8)
        ax.plot(x_axis, predictions_plot[:, i], color=colors_pred[i], 
                label=f'{var} Predicción', linewidth=1, alpha=0.8)
        
        ax.set_ylabel(var)
        ax.set_title(f'{var} - Real vs Predicción')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Tiempo (muestras)')
    plt.tight_layout()
    plt.show()

# Generar gráficos para todas las estaciones
stations_data = [
    ('M5023', y_real_M5023, predictions_M5023),
    ('M5025', y_real_M5025, predictions_M5025),
    ('P34', y_real_P34, predictions_P34),
    ('P63', y_real_P63, predictions_P63)
]

print("\n" + "="*50)
print("GENERANDO GRÁFICOS")
print("="*50)

for station_name, y_real, predictions in stations_data:
    print(f"\nGenerando gráficos para estación {station_name}...")
    plot_predictions_multivariate(y_real, predictions, station_name)
    plot_time_series(y_real, predictions, station_name)

print("\n¡Proceso completado exitosamente!")
print("Se han generado modelos LSTM que predicen las 3 variables meteorológicas:")
print("- Precipitación (mm)")
print("- Temperatura (°C)")  
print("- Humedad Relativa (%)")
