# Importación de las librerías necesarias
import datetime
import json
import configparser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import html
import re
import tiktoken
import os  # Agregado para manejar rutas de forma segura

# Cargar la configuración desde config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Obtener las variables de configuración
try:
    nombre_archivo_json = config['SERVICIOS_SIMAP']['nombre_archivo_json']
    directorio_archivo_json = config['SERVICIOS_SIMAP']['directorio_archivo_json']
    ruta_archivo_json = os.path.join(directorio_archivo_json, nombre_archivo_json)  # Modificado para usar os.path.join
    openai_api_key = config['DEFAULT']['openai_api_key']
    directorio_bdvectorial = config['SERVICIOS_SIMAP']['FRAGMENT_STORE_DIR']
    nombre_bdvectorial = config['SERVICIOS_SIMAP']['nombre_bdvectorial']
    tamano_chunk = int(config['SERVICIOS_SIMAP']['tamano_chunk'])
    overlap_chunk = int(config['SERVICIOS_SIMAP']['overlap_chunk'])
except KeyError as e:
    raise ValueError(f"Falta la clave de configuración: {e}")

# Función para contar tokens usando tiktoken
def contar_tokens(texto):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(texto))

# Normalización de texto
def normalizar_texto(texto):
    if texto is None:
        return ""
    texto_decodificado = html.unescape(texto)
    texto_decodificado = texto_decodificado.replace("\r", " ").replace("\n", " ")
    texto_limpio = re.sub(r'\s+', ' ', texto_decodificado)
    return texto_limpio.strip()

# Función para dividir texto en chunks basados en tokens
def dividir_en_chunks(texto, tamano_chunk, solapamiento):
    palabras = texto.split()
    chunk_actual = []

    for palabra in palabras:
        chunk_actual.append(palabra)
        chunk_texto = ' '.join(chunk_actual)
        if contar_tokens(chunk_texto) > tamano_chunk:
            yield ' '.join(chunk_actual[:-1])
            chunk_actual = chunk_actual[-solapamiento:]

    if chunk_actual:
        yield ' '.join(chunk_actual)

# Función principal para cargar JSON en la base de datos vectorial
def cargar_json_a_chroma(ruta_archivo_json, openai_api_key, directorio_bdvectorial, nombre_bdvectorial, tamano_chunk, solapamiento):
    if not os.path.exists(ruta_archivo_json):  # Validación agregada para verificar si el archivo existe
        raise FileNotFoundError(f"El archivo {ruta_archivo_json} no existe.")

    with open(ruta_archivo_json, 'r', encoding='utf-8') as archivo:
        data = json.load(archivo)

    documentos = []
    registros = data.get("RECORDS", [])
    for item in registros:
        metadata = {
            "servicio": normalizar_texto(item.get("SERVICIO", "")),
            "tipo": normalizar_texto(item.get("TIPO", "")),
            "subtipo": normalizar_texto(item.get("SUBTIPO", "")),
            "id_sub": item.get("ID_SUB", "")
        }

        for campo in ["COPETE", "CONSISTE", "REQUISITOS", "PAUTAS", "QUIEN_PUEDE", "QUIENES_PUEDEN", "COMO_LO_HACEN"]:
            texto = item.get(campo)
            texto_normalizado = normalizar_texto(texto)
            if texto_normalizado:
                texto_con_metadata = f"{campo}: {texto_normalizado}"
                for chunk in dividir_en_chunks(texto_con_metadata, tamano_chunk, solapamiento):
                    documento = Document(page_content=chunk, metadata=metadata)
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
