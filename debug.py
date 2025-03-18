import os
import sys
import configparser
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Función para imprimir separadores
def print_separator(title=""):
    print("\n" + "-" * 40)
    if title:
        print(f"🔹 {title}")
        print("-" * 40)

# Cargar la configuración desde config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Verificar si la sección SERVICIOS_SIMAP existe antes de acceder a ella
if 'SERVICIOS_SIMAP' not in config:
    print("❌ ERROR: La sección 'SERVICIOS_SIMAP' no existe en config.ini. Verifica el nombre.")
    sys.exit(1)

# Obtener la ruta del almacenamiento de fragmentos
ruta_original = 'data/SERVICIOS/SERVICIOS_XX'
collection_name_fragmento = config['DEFAULT'].get('collection_name_fragmento', fallback='fragment_store')

print_separator("Parámetros obtenidos de config.ini")
print(f"📂 Ruta BDV: {ruta_original}")
print(f"📚 Colección: {collection_name_fragmento}")

def verificar_directorio(ruta):
    """ Verifica si el directorio existe y lista los archivos en su interior. """
    print_separator(f"#######################Verificando existencia del directorio: {ruta}")
    print(f"Buscando en el path: {ruta}")
    if os.path.exists(ruta):
        print(f"✅ La carpeta '{ruta}' EXISTE en el sistema.")
        archivos = os.listdir(ruta)
        if archivos:
            print(f"📂 Archivos en '{ruta}':")
            for archivo in archivos:
                print(f"   - {archivo}")
        else:
            print("⚠️ La carpeta está vacía.")
    else:
        print(f"❌ ERROR: La carpeta '{ruta}' NO EXISTE en el sistema.")
        sys.exit(1)

def probar_conexion_chroma(ruta):
    """ Intenta abrir la base de datos vectorial en ChromaDB sin crear una nueva si no existe. """
    print_separator("Probando conexión con ChromaDB")
    print(f"Buscando en el path: {ruta}")
    if not os.path.exists(ruta):
        print(f"❌ ERROR: La BDV en '{ruta}' no existe. No se puede conectar.")
        sys.exit(1)
    try:
        client = PersistentClient(path=ruta)
        collections = client.list_collections()
        if collections:
            print(f"✅ Conexión a BDV exitosa. Colecciones encontradas: {collections}")
        else:
            print("⚠️ Conexión a BDV exitosa, pero no se encontraron colecciones.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR al abrir la base de datos vectorial con ChromaDB: {str(e)}")
        sys.exit(1)

def probar_conexion_langchain(ruta, collection_name):
    """ Intenta abrir la BDV con LangChain y verificar si puede acceder a la colección. """
    print_separator("Probando conexión con LangChain")
    print(f"Buscando en el path: {ruta}")
    if not os.path.exists(ruta):
        print(f"❌ ERROR: La BDV en '{ruta}' no existe. No se puede conectar.")
        sys.exit(1)
    try:
        api_key = config['DEFAULT'].get('openai_api_key', os.environ.get('OPENAI_API_KEY', None))
        if not api_key:
            print("❌ ERROR: No se encontró la API Key de OpenAI.")
            sys.exit(1)
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=ruta,
            embedding_function=embeddings
        )

        # Verificar si tiene documentos
        docs = vector_store.similarity_search("test", k=1)
        if docs:
            print("✅ Conexión a BDV con LangChain exitosa. Se encontraron documentos.")
        else:
            print("⚠️ Conexión a BDV con LangChain exitosa, pero no hay documentos almacenados.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR al abrir la base de datos vectorial con LangChain: {str(e)}")
        sys.exit(1)

# Ejecutar las verificaciones con la ruta de config.ini
#verificar_directorio(ruta_original)
probar_conexion_chroma(ruta_original)
probar_conexion_langchain(ruta_original, collection_name_fragmento)

# Si se pasa un argumento en la línea de comandos, probar con él
if len(sys.argv) > 1:
    ruta_alternativa = sys.argv[1]
    verificar_directorio(ruta_alternativa)
    probar_conexion_chroma(ruta_alternativa)
    probar_conexion_langchain(ruta_alternativa, collection_name_fragmento)
