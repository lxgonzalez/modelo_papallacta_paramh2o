#!/usr/bin/env python3
"""
Script de prueba para verificar el análisis conciso.
"""

import requests
import json

def test_analysis():
    """Prueba la API con análisis conciso"""
    
    url = "http://localhost:5000/predict"
    
    data = {
        "date": "2024-06-01",
        "latitude": -0.35,
        "longitude": -78.17,
        "include_analysis": True,
        "analysis_types": ["general", "cultivos", "riego"]
    }
    
    print("🧪 Probando análisis conciso...")
    print(f"📍 Ubicación: {data['latitude']}, {data['longitude']}")
    print(f"📅 Fecha: {data['date']}")
    print(f"🔍 Análisis: {data['analysis_types']}")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success') and 'agricultural_analysis' in result.get('data', {}):
                analysis = result['data']['agricultural_analysis']
                
                print("✅ Análisis obtenido:")
                print(json.dumps(analysis, indent=2, ensure_ascii=False))
                
                # Verificar longitud de respuestas
                print("\n📊 Verificación de longitud:")
                for key, value in analysis.items():
                    if isinstance(value, str):
                        print(f"  {key}: {len(value)} caracteres")
                        if len(value) > 300:
                            print(f"    ⚠️  Demasiado largo (>300 chars)")
                        else:
                            print(f"    ✅ Longitud apropiada")
                    elif isinstance(value, list):
                        print(f"  {key}: {len(value)} elementos")
                        if len(value) > 7:
                            print(f"    ⚠️  Demasiados elementos (>7)")
                        else:
                            print(f"    ✅ Cantidad apropiada")
                
            else:
                print("❌ No se encontró análisis en la respuesta")
                print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Error HTTP {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se pudo conectar al servidor")
        print("💡 Asegúrate de que la API esté ejecutándose en http://localhost:5000")
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")

if __name__ == "__main__":
    test_analysis()
