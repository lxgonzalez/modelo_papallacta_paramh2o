import pandas as pd

df = pd.read_csv("Precipitación_Horario__Papallacta.csv")
df['Fecha'] = pd.to_datetime(df['Fecha'])

fecha_inicio = pd.to_datetime("2014-06-23 10:00:00")
clean_df = df[df['Fecha'] >= fecha_inicio].copy()

for col in ['P34', 'P63', 'M5023', 'M5025']:
    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

clean_df.dropna(subset=['P34', 'P63', 'M5023', 'M5025'], how='all', inplace=True)

clean_df.to_csv("precipitacion_limpia.csv", index=False)

print("Archivo limpio guardado como precipitacion_limpia.csv")
