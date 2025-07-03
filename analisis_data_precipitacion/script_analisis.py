import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuración inicial
csv_input = "precipitacion_limpia.csv"
csv_output = "precipitacion_features.csv"
folder_graficas = "graficas"
txt_output = "estadisticas.txt"

if not os.path.exists(folder_graficas):
    os.makedirs(folder_graficas)

# Cargar datos
df = pd.read_csv(csv_input)
df['Fecha'] = pd.to_datetime(df['Fecha'])
df.set_index('Fecha', inplace=True)

# Columnas de estaciones
estaciones = ['P34', 'P63', 'M5023', 'M5025']

# 1. Graficar series de tiempo y guardar gráficos
for est in estaciones:
    plt.figure(figsize=(12, 4))
    df[est].plot(title=f"Serie de tiempo - Estación {est}")
    plt.xlabel("Fecha")
    plt.ylabel("Precipitación (mm)")
    plt.tight_layout()
    plt.savefig(f"{folder_graficas}/serie_tiempo_{est}.png")
    plt.close()

# 2. Calcular estadísticas descriptivas y guardar en txt
with open(txt_output, 'w') as f:
    f.write("Estadísticas descriptivas de precipitaciones\n")
    f.write("==========================================\n\n")
    for est in estaciones:
        f.write(f"Estación: {est}\n")
        f.write(df[est].describe().to_string())
        f.write("\n\n")

# 3. Matriz de correlación y gráfica
corr = df[estaciones].corr()
plt.figure(figsize=(8, 6))
plt.title("Matriz de correlación entre estaciones")
cax = plt.matshow(corr, fignum=1)
plt.colorbar(cax)
plt.xticks(range(len(estaciones)), estaciones)
plt.yticks(range(len(estaciones)), estaciones)
plt.savefig(f"{folder_graficas}/matriz_correlacion.png")
plt.close()

# Guardar matriz de correlación en txt
with open(txt_output, 'a') as f:
    f.write("Matriz de correlación entre estaciones:\n")
    f.write(corr.to_string())
    f.write("\n\n")

# 4. Crear features temporales (lags y medias móviles)
lags = [1, 3, 6, 12]  # Horas de desfase para crear features
for est in estaciones:
    for lag in lags:
        df[f'{est}_lag_{lag}'] = df[est].shift(lag)
    # Medias móviles a 3 y 6 horas
    df[f'{est}_rolling_3'] = df[est].rolling(window=3).mean()
    df[f'{est}_rolling_6'] = df[est].rolling(window=6).mean()

# Guardar CSV con features
df.to_csv(csv_output)

# 5. Detección de eventos extremos (picos y mínimos)
# Definimos umbral como percentil 95 para picos, percentil 5 para mínimos
with open(txt_output, 'a') as f:
    f.write("Eventos extremos detectados (picos y mínimos):\n")
    f.write("=============================================\n\n")
    
    for est in estaciones:
        p95 = df[est].quantile(0.95)
        p05 = df[est].quantile(0.05)
        
        picos = df[df[est] >= p95][est]
        minimos = df[df[est] <= p05][est]
        
        f.write(f"Estación {est} - Umbral pico (95%): {p95:.3f}\n")
        f.write(f"Estación {est} - Umbral mínimo (5%): {p05:.3f}\n")
        f.write(f"Cantidad de picos: {len(picos)}\n")
        f.write(f"Cantidad de mínimos: {len(minimos)}\n\n")
        
        # Graficar picos
        plt.figure(figsize=(12, 4))
        df[est].plot(label='Precipitación')
        plt.scatter(picos.index, picos.values, color='red', label='Picos (95%)')
        plt.scatter(minimos.index, minimos.values, color='blue', label='Mínimos (5%)')
        plt.title(f"Eventos extremos - Estación {est}")
        plt.xlabel("Fecha")
        plt.ylabel("Precipitación (mm)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder_graficas}/eventos_extremos_{est}.png")
        plt.close()
