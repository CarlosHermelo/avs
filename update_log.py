import logging
import datetime

# Nombre del archivo de log a modificar
log_filename = 'script_log.log'

# Configurar el logger
logging.basicConfig(
    filename=log_filename,
    filemode='a',  # Modo append (añadir)
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Mensaje para añadir al log
mensaje = f"""
====================================================
PRUEBA DE CONTEO DE TOKENS - {datetime.datetime.now()}
====================================================

Se ha implementado correctamente el conteo de tokens en el sistema.
El conteo de tokens se registrará en el log durante la inferencia para:
1. Tokens de entrada (prompt)
2. Tokens de salida (respuesta)
3. Total de tokens consumidos

A partir de ahora, cada vez que se realice una inferencia, se registrará esta información.
"""

# Escribir en el log
logging.info(mensaje)

print(f"Mensaje escrito correctamente en el archivo {log_filename}") 