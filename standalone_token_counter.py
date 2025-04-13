"""
Script independiente para demostrar el formato de registro de tokens 
que se usará en el archivo principal grafo_AGENTE_SERV_flask.py
"""

import logging
import datetime
import json
import tiktoken

# Configuración del logging
log_filename = 'token_test_log.log'
logging.basicConfig(
    filename=log_filename,
    filemode='a',  # Agregar registros al final del archivo existente
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def log_message(message, level='INFO'):
    """Registra un mensaje en el log."""
    if level.upper() == 'INFO':
        logging.info(message)
    elif level.upper() == 'WARNING':
        logging.warning(message)
    elif level.upper() == 'ERROR':
        logging.error(message)
    elif level.upper() == 'DEBUG':
        logging.debug(message)
    else:
        logging.info(message)
    # También mostrar en consola
    print(message)

def contar_tokens(texto, modelo="gpt-3.5-turbo"):
    """
    Cuenta el número de tokens en un texto para un modelo específico.
    
    Args:
        texto (str): El texto para contar tokens
        modelo (str): El nombre del modelo (por defecto: gpt-3.5-turbo)
        
    Returns:
        int: Número de tokens
    """
    try:
        # Mapeamos nombres de modelos a codificadores
        if modelo.startswith("gpt-4"):
            codificador = tiktoken.encoding_for_model("gpt-4")
        elif modelo.startswith("gpt-3.5"):
            codificador = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            # Usamos cl100k_base para modelos más recientes
            codificador = tiktoken.get_encoding("cl100k_base")
            
        # Contar tokens
        tokens = len(codificador.encode(texto))
        return tokens
    except Exception as e:
        log_message(f"Error al contar tokens: {str(e)}", level='ERROR')
        return 0

def log_token_summary(tokens_entrada, tokens_salida, modelo):
    """
    Registra un resumen claro del conteo de tokens para cada inferencia.
    
    Args:
        tokens_entrada (int): Número de tokens de la entrada (pregunta + contexto)
        tokens_salida (int): Número de tokens de la respuesta
        modelo (str): Nombre del modelo utilizado
    """
    separador = "=" * 80
    log_message(separador)
    log_message("RESUMEN DE CONTEO DE TOKENS")
    log_message(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Modelo: {modelo}")
    log_message(separador)
    log_message(f"TOKENS DE ENTRADA (pregunta + contexto): {tokens_entrada}")
    log_message(f"TOKENS DE SALIDA (respuesta final): {tokens_salida}")
    log_message(f"TOTAL TOKENS CONSUMIDOS: {tokens_entrada + tokens_salida}")
    
    # Calcular costo aproximado (solo para referencia)
    costo_aprox = 0
    
    if modelo.startswith("gpt-4"):
        costo_entrada = round((tokens_entrada / 1000) * 0.03, 4)  # $0.03 por 1K tokens entrada
        costo_salida = round((tokens_salida / 1000) * 0.06, 4)    # $0.06 por 1K tokens salida
        costo_aprox = costo_entrada + costo_salida
    elif modelo.startswith("gpt-3.5"):
        costo_aprox = round(((tokens_entrada + tokens_salida) / 1000) * 0.002, 4)  # $0.002 por 1K tokens
    
    log_message(f"COSTO APROXIMADO USD: ${costo_aprox}")
    log_message(separador)
    
    # Guardar en formato JSON para facilitar análisis posterior
    resumen_json = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": modelo,
        "input_tokens": tokens_entrada,
        "output_tokens": tokens_salida,
        "total_tokens": tokens_entrada + tokens_salida,
        "approx_cost_usd": costo_aprox
    }
    
    log_message(f"RESUMEN_JSON: {json.dumps(resumen_json)}")
    log_message(separador)

if __name__ == "__main__":
    # Ejemplo de uso
    print("Demostrando el formato de registro de tokens")
    
    # Texto de ejemplo
    pregunta = "¿Qué servicios ofrece PAMI para personas mayores?"
    contexto = """
    PAMI ofrece varios servicios para personas mayores:
    - Bolsón alimentario
    - Apoyo a la dependencia y fragilidad
    - Viviendas en comodato
    - Atención médica domiciliaria
    - Club de día para personas mayores
    
    Se requiere documentación específica para cada servicio:
    - DNI
    - Credencial de afiliación
    - Informes médicos (según corresponda)
    """
    
    prompt_completo = pregunta + "\n\n" + contexto
    
    respuesta = """
    PAMI ofrece los siguientes servicios para personas mayores:
    
    1. Bolsón alimentario: Entrega mensual de productos secos seleccionados por nutricionistas.
    2. Apoyo a la Dependencia y Fragilidad (PADyF): Apoyo económico para personas con limitaciones funcionales.
    3. Viviendas en comodato: Otorgamiento de vivienda con descuento del 10% del haber previsional.
    4. Club de día: Prestación socio comunitaria con jornadas de 6-8 horas diarias de actividades.
    
    Requisitos generales:
    - DNI
    - Credencial de afiliación
    - Evaluación del Equipo Social
    
    Referencias:
    - ID_SUB = 66 | SUBTIPO = 'Bolsón Alimentario-Probienestar-BCA'
    - ID_SUB = 43 | SUBTIPO = 'Apoyo a la Dependencia y Fragilidad- PADyF'
    """
    
    # Calcular tokens
    tokens_entrada = contar_tokens(prompt_completo)
    tokens_salida = contar_tokens(respuesta)
    
    # Mostrar resultados
    print(f"\nTexto de entrada (pregunta + contexto): {len(prompt_completo)} caracteres")
    print(f"Texto de salida (respuesta): {len(respuesta)} caracteres")
    print(f"\nTokens de entrada: {tokens_entrada}")
    print(f"Tokens de salida: {tokens_salida}")
    print(f"Total tokens: {tokens_entrada + tokens_salida}")
    
    # Registrar en el log el formato que se usará
    log_message("\nEjemplo de registro de tokens para grafo_AGENTE_SERV_flask.py:")
    log_token_summary(tokens_entrada, tokens_salida, "gpt-4o-mini")
    
    print(f"\nLos resultados han sido registrados en {log_filename}")
    print("Este es el formato que se implementará en grafo_AGENTE_SERV_flask.py") 