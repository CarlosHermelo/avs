# Importación de las librerías necesarias
import datetime
import json
import configparser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import html
import re

# Cargar la configuración desde config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Obtener las variables de configuración
try:
    nombre_archivo_json = config['SERVICIOS_SIMAP']['nombre_archivo_json']
    directorio_archivo_json = config['SERVICIOS_SIMAP']['directorio_archivo_json']
    ruta_archivo_json = f"{directorio_archivo_json}/{nombre_archivo_json}"
    openai_api_key = config['DEFAULT']['openai_api_key']
    directorio_bdvectorial = config['SERVICIOS_SIMAP']['FRAGMENT_STORE_DIR']
    nombre_bdvectorial = config['SERVICIOS_SIMAP']['nombre_bdvectorial']
    tamano_chunk = int(config['SERVICIOS_SIMAP']['tamano_chunk'])
    overlap_chunk = int(config['SERVICIOS_SIMAP']['overlap_chunk'])
except KeyError as e:
    raise ValueError(f"Falta la clave de configuración: {e}")


def normalizar_texto(texto):
    if texto is None:
        return ""
    
    # 1. Decodificar entidades HTML
    texto_decodificado = html.unescape(texto)
    
    # 2. Reemplazar saltos de línea y espacios redundantes
    texto_decodificado = texto_decodificado.replace("\r", " ").replace("\n", " ")
    
    # 3. Reemplazar múltiples espacios por uno solo
    texto_limpio = re.sub(r'\s+', ' ', texto_decodificado)
    
    # 4. Eliminar espacios al inicio y al final
    return texto_limpio.strip()

def conservar_urls_emails(texto):
    """
    Normaliza el texto pero conserva las URLs y emails.
    """
    if texto is None:
        return ""
    
    # 1. Detectar URLs y emails usando expresiones regulares
    urls_emails = re.findall(r'(https?://\S+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', texto)
    
    # 2. Normalizar el texto (remover entidades HTML, espacios redundantes, etc.)
    texto_normalizado = normalizar_texto(texto)
    
    # 3. Reinsertar las URLs y emails detectados en el texto normalizado
    for url_email in urls_emails:
        if url_email not in texto_normalizado:
            texto_normalizado = texto_normalizado.replace(url_email.replace(" ", ""), url_email)
    
    return texto_normalizado.strip()
# Función para normalizar texto

# Función para dividir texto en chunks
def dividir_en_chunks(texto, tamano_chunk, solapamiento):
    palabras = texto.split()
    if tamano_chunk <= solapamiento:
        raise ValueError("El tamaño del chunk debe ser mayor que el solapamiento.")

    for i in range(0, len(palabras), tamano_chunk - solapamiento):
        chunk = ' '.join(palabras[i:i + tamano_chunk])
        yield chunk

# Función principal para cargar JSON en la base de datos vectorial
def cargar_json_a_chroma(ruta_archivo_json, openai_api_key, directorio_bdvectorial, nombre_bdvectorial, tamano_chunk, solapamiento):
    with open(ruta_archivo_json, 'r', encoding='utf-8') as archivo:
        data = json.load(archivo)

    documentos = []
    registros = data.get("RECORDS", [])
    for item in registros:
        servicio = normalizar_texto(item.get("SERVICIO", ""))
        tipo = normalizar_texto(item.get("TIPO", ""))
        subtipo = normalizar_texto(item.get("SUBTIPO", ""))
        id_sub = item.get("ID_SUB", "")

        for campo in ["COPETE", "CONSISTE", "REQUISITOS", "PAUTAS", "QUIEN_PUEDE", "QUIENES_PUEDEN", "COMO_LO_HACEN"]:
            texto = item.get(campo)
            texto_normalizado = normalizar_texto(texto)
            if texto_normalizado:
                texto_con_metadata = (
                    f"[SERVICIO: {servicio}] [TIPO: {tipo}] [SUBTIPO: {subtipo}] [ID_SUB: {id_sub}]   {campo}:"
                    f"{texto_normalizado}"
                )
                for chunk in dividir_en_chunks(texto_con_metadata, tamano_chunk, solapamiento):
                    documento = Document(
                        page_content=chunk,
                        metadata={
                            "servicio": servicio,
                            "tipo": tipo,
                            "subtipo": subtipo,
                            "id_sub": id_sub,
                            "campo": campo
                        }
                    )
                    documentos.append(documento)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory=directorio_bdvectorial,
        collection_name=nombre_bdvectorial
    )
    print(f"Se han cargado {len(documentos)} documentos en la base de datos vectorial.")

# Llamada de ejemplo a la función
cargar_json_a_chroma(ruta_archivo_json, openai_api_key, directorio_bdvectorial, nombre_bdvectorial, tamano_chunk, overlap_chunk)
