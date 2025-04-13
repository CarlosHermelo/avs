"""
Script para probar la inferencia y registro de conteo de tokens.
Este script llama directamente a la función process_question del módulo grafo_AGENTE_SERV_flask.py
sin depender de la aplicación Flask completa.
"""

from grafo_AGENTE_SERV_flask import process_question

# Definir una pregunta de prueba
pregunta = "¿Qué servicios ofrece PAMI para personas mayores con movilidad reducida?"
fecha_desde = "2024-01-01"
fecha_hasta = "2024-12-31"
k = 5  # Limitamos a 5 resultados para hacer la prueba más rápida

# Registrar el inicio de la prueba
print("="*80)
print("INICIANDO PRUEBA DE INFERENCIA")
print(f"Pregunta: {pregunta}")
print("="*80)

# Ejecutar la inferencia
try:
    respuesta = process_question(pregunta, fecha_desde, fecha_hasta, k)
    
    # Mostrar la respuesta
    print("\nRESPUESTA OBTENIDA:")
    print("-"*40)
    print(respuesta)
    print("-"*40)
    
    print("\nLa prueba se completó correctamente.")
    print("Ver el archivo script_log.log para los detalles del conteo de tokens.")
    
except Exception as e:
    print(f"\nERROR durante la prueba: {str(e)}")

print("="*80) 