"""
Script para contar tokens de entrada y salida y registrarlos en el log.
Este script permite a los usuarios probar manualmente el conteo de tokens
sin necesidad de ejecutar la aplicación completa.
"""

import tiktoken
import logging
import datetime
import json

# Configuración del logging
log_filename = 'script_log.log'
logging.basicConfig(
    filename=log_filename,
    filemode='a',  # Agregar registros al final del archivo existente
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def log_message(message, level='INFO'):
    """Registra un mensaje en el archivo de log."""
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
    # También imprimimos en consola para ver resultados inmediatos
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

def simular_inferencia(prompt_text, response_text, modelo="gpt-4o-mini"):
    """
    Simula una inferencia y registra el conteo de tokens.
    
    Args:
        prompt_text (str): El texto del prompt
        response_text (str): El texto de la respuesta
        modelo (str): El nombre del modelo
    """
    # Registramos el inicio de la simulación
    log_message(f"\n{'='*50}")
    log_message(f"SIMULACIÓN DE INFERENCIA - {datetime.datetime.now()}")
    log_message(f"Modelo: {modelo}")
    log_message(f"{'='*50}")
    
    # Contamos tokens del prompt
    tokens_prompt = contar_tokens(prompt_text, modelo)
    log_message(f"Tokens de entrada (prompt): {tokens_prompt}")
    
    # Contamos tokens de la respuesta
    tokens_respuesta = contar_tokens(response_text, modelo)
    log_message(f"Tokens de salida (respuesta): {tokens_respuesta}")
    
    # Total de tokens
    total_tokens = tokens_prompt + tokens_respuesta
    log_message(f"Total de tokens consumidos: {total_tokens}")
    
    # Guardamos el resumen en formato JSON para facilitar el procesamiento
    resumen = {
        "timestamp": str(datetime.datetime.now()),
        "modelo": modelo,
        "prompt_tokens": tokens_prompt,
        "response_tokens": tokens_respuesta,
        "total_tokens": total_tokens
    }
    log_message(f"RESUMEN JSON: {json.dumps(resumen)}")
    
    log_message(f"{'='*50}\n")
    
    return {
        "prompt_tokens": tokens_prompt,
        "response_tokens": tokens_respuesta,
        "total_tokens": total_tokens
    }

if __name__ == "__main__":
    # Ejemplo de uso
    prompt_ejemplo = """
    <CONTEXTO>
    La información proporcionada tiene como objetivo apoyar a los agentes que trabajan en las agencias de PAMI,
    quienes se encargan de atender las consultas de los afiliados. Este soporte está diseñado para optimizar
    la experiencia de atención al público y garantizar que los afiliados reciban información confiable y relevante
    en el menor tiempo posible.
    </CONTEXTO>

    <ROL>
    Eres un asistente virtual experto en los servicios y trámites de PAMI.
    </ROL>

    <PREGUNTA>
    ¿Qué servicios ofrece PAMI para personas mayores con movilidad reducida?
    </PREGUNTA>
    """
    
    respuesta_ejemplo = """
    PAMI ofrece varios servicios para personas mayores con movilidad reducida:

    1. Programa de Atención a la Dependencia y Fragilidad (PADyF): Brinda apoyo económico parcial para personas con limitaciones funcionales psicofísicas.

    2. Viviendas en comodato: Otorgamiento de viviendas adaptadas, con descuento del 10% del haber previsional para mantenimiento.

    3. Club de día: Prestación socio comunitaria para personas mayores con autonomía funcional disminuida, con jornadas de 6-8 horas diarias.

    4. Elementos de ayuda técnica: Cobertura para sillas de ruedas, bastones, andadores y otros elementos.

    5. Atención domiciliaria: Servicios médicos y de enfermería a domicilio.

    Requisitos generales:
    - Documento Nacional de Identidad
    - Credencial de afiliación
    - Informes médicos específicos según el servicio

    Referencias:
    - ID_SUB = 43 | SUBTIPO = 'Apoyo a la Dependencia y Fragilidad- PADyF'
    - ID_SUB = 62 | SUBTIPO = 'Viviendas en comodato a través de Programa Pami - Barrios Propios'
    - ID_SUB = 398 | SUBTIPO = 'Club de día para personas mayores'
    """
    
    print("Iniciando simulación de inferencia...")
    resultados = simular_inferencia(prompt_ejemplo, respuesta_ejemplo)
    
    print("\nResultados:")
    print(f"Tokens de entrada: {resultados['prompt_tokens']}")
    print(f"Tokens de salida: {resultados['response_tokens']}")
    print(f"Total de tokens: {resultados['total_tokens']}")
    
    print(f"\nLos resultados han sido registrados en {log_filename}")
    
    # Solicitar al usuario que pruebe con su propio texto
    print("\n¿Desea probar con su propio texto? (s/n)")
    respuesta = input("> ")
    
    if respuesta.lower() == 's':
        print("\nIntroduzca el texto del prompt (terminar con 'EOF' en una línea):")
        lineas_prompt = []
        while True:
            linea = input()
            if linea == 'EOF':
                break
            lineas_prompt.append(linea)
        
        prompt_usuario = "\n".join(lineas_prompt)
        
        print("\nIntroduzca el texto de la respuesta (terminar con 'EOF' en una línea):")
        lineas_respuesta = []
        while True:
            linea = input()
            if linea == 'EOF':
                break
            lineas_respuesta.append(linea)
        
        respuesta_usuario = "\n".join(lineas_respuesta)
        
        print("\nIntroduzca el nombre del modelo (por defecto: gpt-4o-mini):")
        modelo_usuario = input("> ")
        if not modelo_usuario:
            modelo_usuario = "gpt-4o-mini"
        
        print("\nProcesando...")
        resultados_usuario = simular_inferencia(prompt_usuario, respuesta_usuario, modelo_usuario)
        
        print("\nResultados:")
        print(f"Tokens de entrada: {resultados_usuario['prompt_tokens']}")
        print(f"Tokens de salida: {resultados_usuario['response_tokens']}")
        print(f"Total de tokens: {resultados_usuario['total_tokens']}")
        
        print(f"\nLos resultados han sido registrados en {log_filename}")
    
    print("\n¡Gracias por usar el contador de tokens!") 