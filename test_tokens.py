import tiktoken
import logging
import os

# Configuración del logging
logging.basicConfig(
    filename='token_test_log.log',
    filemode='a',  # Agregar registros al final del archivo existente
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Nivel de logging
)

def log_message(message, level='INFO'):
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

def main():
    # Prueba con diferentes textos
    log_message("Iniciando prueba de conteo de tokens")
    
    # Texto simple
    texto_simple = "Este es un texto simple para contar tokens"
    tokens_simple = contar_tokens(texto_simple)
    log_message(f"Texto simple ({len(texto_simple)} caracteres): {tokens_simple} tokens")
    
    # Texto más largo
    texto_largo = "Este es un texto más largo para verificar cómo funciona el conteo de tokens en textos de mayor longitud. " * 10
    tokens_largo = contar_tokens(texto_largo)
    log_message(f"Texto largo ({len(texto_largo)} caracteres): {tokens_largo} tokens")
    
    # Texto con caracteres especiales
    texto_especial = "Este texto tiene caracteres especiales: åäö üß ñ 你好 😊"
    tokens_especial = contar_tokens(texto_especial)
    log_message(f"Texto con caracteres especiales ({len(texto_especial)} caracteres): {tokens_especial} tokens")
    
    # Diferentes modelos
    log_message(f"Conteo con diferentes modelos (texto simple):")
    log_message(f"GPT-3.5-turbo: {contar_tokens(texto_simple, 'gpt-3.5-turbo')}")
    log_message(f"GPT-4: {contar_tokens(texto_simple, 'gpt-4')}")
    
    log_message("Prueba de conteo de tokens finalizada")

if __name__ == "__main__":
    main() 