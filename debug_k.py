import os
import sys
import configparser
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Cargar la configuración desde config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Obtener la API Key desde config.ini o variable de entorno
API_KEY = config['DEFAULT'].get('openai_api_key', os.environ.get('OPENAI_API_KEY', None))
if not API_KEY:
    print("❌ ERROR: No se encontró la API Key de OpenAI en config.ini ni en variables de entorno.")
    sys.exit(1)

# Definir el path de la BDV
BDV_PATH = "data/SERVICIOS/SERVICIOS_XX"
COLLECTION_NAME = "servicios_collection"

# Función para imprimir separadores
def print_separator(title=""):
    print("\n" + "-" * 40)
    if title:
        print(f"🔹 {title}")
        print("-" * 40)

# Verificar si el directorio existe, si no crearlo
def verificar_o_crear_directorio(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)
        print(f"✅ Directorio '{ruta}' creado exitosamente.")
    else:
        print(f"✅ El directorio '{ruta}' ya existe.")

# Crear BDV con dos registros
def crear_bdv():
    print_separator("Creando BDV y agregando registros")
    verificar_o_crear_directorio(BDV_PATH)
    
    try:
        embeddings = OpenAIEmbeddings(api_key=API_KEY)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=BDV_PATH,
            embedding_function=embeddings
        )
        
        docs = [
            Document(page_content="Primer documento de prueba."),
            Document(page_content="Segundo documento de prueba.")
        ]
        
        vector_store.add_documents(docs)
        print("✅ BDV creada y registros agregados correctamente.")
    
    except Exception as e:
        print(f"❌ ERROR al crear la BDV: {str(e)}")
        sys.exit(1)

# Probar la conexión con la BDV existente
def probar_conexion_langchain():
    print_separator("Probando conexión con BDV existente")
    if not os.path.exists(BDV_PATH):
        print(f"❌ ERROR: La BDV en '{BDV_PATH}' no existe. No se puede conectar.")
        sys.exit(1)
    
    try:
        embeddings = OpenAIEmbeddings(api_key=API_KEY)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=BDV_PATH,
            embedding_function=embeddings
        )
        
        docs = vector_store.similarity_search("prueba", k=2)
        if docs:
            print("✅ Conexión a BDV con LangChain exitosa. Se encontraron documentos:")
            for doc in docs:
                print(f"   - {doc.page_content}")
        else:
            print("⚠️ Conexión a BDV con LangChain exitosa, pero no hay documentos almacenados.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR al abrir la base de datos vectorial con LangChain: {str(e)}")
        sys.exit(1)

# Ejecutar el proceso
crear_bdv()
probar_conexion_langchain()
