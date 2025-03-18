import json
import configparser
import sqlite3
import os
import html
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

# Cargar configuración desde config.ini
config = configparser.ConfigParser()
config.read('config.ini')

try:
    model_name = config['DEFAULT'].get('modelo')
    openai_api_key = config['DEFAULT'].get('openai_api_key')
    directorio_bdvectorial = config['SERVICIOS_SIMAP'].get('FRAGMENT_STORE_DIR', './chroma_db')
    nombre_bdvectorial = config['SERVICIOS_SIMAP'].get('nombre_bdvectorial', 'vector_db')
    tamano_chunk = int(config['SERVICIOS_SIMAP'].get('tamano_chunk', 100))
    overlap_chunk = int(config['SERVICIOS_SIMAP'].get('overlap_chunk', 10))
    ruta_archivo_json = os.path.join(config['SERVICIOS_SIMAP'].get('directorio_archivo_json', '.'), 
                                     config['SERVICIOS_SIMAP'].get('nombre_archivo_json', 'data.json'))
    base_datos_bm25 = config['SERVICIOS_SIMAP'].get('BM25_DB_PATH', 'bm25_index.db')
except KeyError as e:
    raise ValueError(f"Error en la configuración: Falta la clave {e}")

llm = ChatOpenAI(api_key=openai_api_key, model_name=model_name)

prompt_template = PromptTemplate(
    input_variables=["whole_document", "fragment"],
    template="""
    <full_document>
    {whole_document}
    </full_document>

    <current_chunk>
    {fragment}
    </current_chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. Escribilo en castellano y solo 80 palabras.
    """
)
llm_chain = prompt_template | llm

def normalizar_texto(texto):
    if not texto:
        return ""
    texto = html.unescape(texto).replace("\r", " ").replace("\n", " ")
    return re.sub(r'\s+', ' ', texto).strip()

def dividir_en_chunks(texto, tamano_chunk, solapamiento):
    palabras = texto.split()
    paso = tamano_chunk - solapamiento
    return [' '.join(palabras[i:i + tamano_chunk]) for i in range(0, len(palabras), paso)]

def procesar_json_y_cargar_bd():
    print("Iniciando procesamiento de JSON y carga en base de datos...")
    
    if not os.path.exists(ruta_archivo_json):
        raise FileNotFoundError(f"No se encontró el archivo JSON en la ruta: {ruta_archivo_json}")
    
    with open(ruta_archivo_json, 'r', encoding='utf-8') as archivo:
        data = json.load(archivo)

    registros = data.get("RECORDS", [])
    documentos = []
    
    conn_bm25 = sqlite3.connect(base_datos_bm25)
    c = conn_bm25.cursor()
    c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(chunk_content)")
    
    clave_actual = ""
    contenido_tipo = ""
    servicio_anterior, tipo_anterior, subtipo_anterior, id_sub_anterior = "", "", "", ""
    
    for item in registros:
        servicio = normalizar_texto(item.get("SERVICIO", ""))
        tipo = normalizar_texto(item.get("TIPO", ""))
        subtipo = normalizar_texto(item.get("SUBTIPO", ""))
        id_sub = normalizar_texto(str(item.get("ID_SUB", "")))
        
        clave_nueva = f"{servicio}|{tipo}|{subtipo}|{id_sub}"
        
        if clave_nueva != clave_actual and clave_actual:
            print(f"Corte de control - Whole Document: {contenido_tipo}\n\n")
            chunks = dividir_en_chunks(contenido_tipo, tamano_chunk, overlap_chunk)
            for chunk in chunks:
                contexto_actual = llm_chain.invoke({
                    "whole_document": contenido_tipo,
                    "fragment": chunk
                }).content
                
                fragmento_completo = (
                    f"<contexto>{contexto_actual}</contexto> "
                    f"<fragmento>[SERVICIO: {servicio_anterior}] [TIPO: {tipo_anterior}] [SUBTIPO: {subtipo_anterior}] [ID_SUB: {id_sub_anterior}] {chunk}</fragmento>"
                )
                
                print(f"Registro BDVectorial: {fragmento_completo}\n\n")
                documentos.append(fragmento_completo)
                c.execute("INSERT INTO chunks VALUES (?)", (fragmento_completo,))
            
            contenido_tipo = ""
        
        servicio_anterior, tipo_anterior, subtipo_anterior, id_sub_anterior = servicio, tipo, subtipo, id_sub
        contenido_tipo += f" SERVICIO: {servicio} TIPO: {tipo} SUBTIPO: {subtipo} ID_SUB: {id_sub}"
        
        campos = ["COPETE", "CONSISTE", "REQUISITOS", "PAUTAS", "QUIEN_PUEDE", "QUIENES_PUEDEN", "COMO_LO_HACEN"]
        for campo in campos:
            texto = normalizar_texto(item.get(campo, ""))
            if texto:
                contenido_tipo += f" {campo}: {texto}"
        
        clave_actual = clave_nueva
    
    conn_bm25.commit()
    conn_bm25.close()
    
    print("Generando embeddings y cargando en ChromaDB...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = Chroma.from_texts(
        texts=documentos,
        embedding=embeddings,
        persist_directory=directorio_bdvectorial,
        collection_name=nombre_bdvectorial
    )
    
    print(f"Se han cargado {len(documentos)} documentos en ChromaDB y en BM25.")

procesar_json_y_cargar_bd()
