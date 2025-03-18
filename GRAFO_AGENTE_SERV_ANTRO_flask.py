# 🛠 Configuración e Instalación de Dependencias

import os
import getpass
import configparser
import logging
import sqlite3
import re
import numpy as np
import cohere

# --------------------- Configuración de Logging ---------------------
glog_filename = 'script_log_antro.log'  # Nombre del archivo de log
logging.basicConfig(
    filename=glog_filename,
    filemode='a',  # Agregar registros al final del archivo existente
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Nivel de logging (ajustable según necesidad)
)

def log_message(message, level='DEBUG'):
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

# --------------------- Cargar Configuración ---------------------
config = configparser.ConfigParser()
config.read('config.ini')

# Variables originales
collection_name_fragmento = config['DEFAULT'].get('collection_name_fragmento', fallback='fragment_store')
model_name = config['DEFAULT'].get('modelo')
log_message(f"Modelo configurado: {model_name}")

fragment_store_directory = config['SERVICIOS_SIMAP_ANTRO'].get('FRAGMENT_STORE_DIR', fallback='/content/chroma_fragment_store')
# Nota: Para la búsqueda semántica se usará 'max_results_chroma' (nueva variable)
max_results_chroma = config['SERVICIOS_SIMAP_ANTRO'].getint('max_results_chroma', fallback=50)
fecha_desde_pagina = config['SERVICIOS_SIMAP_ANTRO'].get('fecha_desde', fallback='2024-01-08')
fecha_hasta_pagina = config['SERVICIOS_SIMAP_ANTRO'].get('fecha_hasta', fallback='2024-12-10')

log_message(f"Fragment store directory: {fragment_store_directory}")
log_message(f"Max results (Chroma) configurado: {max_results_chroma}")
log_message(f"Rango de fechas configurado: {fecha_desde_pagina} a {fecha_hasta_pagina}")

# Variables para BM25 y reranking
bm25_db_path = config['SERVICIOS_SIMAP_ANTRO'].get('BM25_DB_PATH', 'bm25_index.db')
max_results_bm25 = config['SERVICIOS_SIMAP_ANTRO'].getint('max_results_bm25', fallback=100)
rerank_enabled = config['SERVICIOS_SIMAP_ANTRO'].getboolean('rerank_enabled', fallback=False)
rerank_top_n = config['SERVICIOS_SIMAP_ANTRO'].getint('rerank_top_n', fallback=150)
rerank_top_k = config['SERVICIOS_SIMAP_ANTRO'].getint('rerank_top_k', fallback=20)

# Configuración de API Keys
api_key = config['DEFAULT'].get('openai_api_key')
os.environ['OPENAI_API_KEY'] = api_key
log_message("API Key OpenAI configurada.")

cohere_api_key = config['DEFAULT'].get('cohere_api_key', '').strip()
if cohere_api_key:
    co = cohere.Client(cohere_api_key)
    log_message("API Key Cohere configurada.")
else:
    co = None
    log_message("API Key Cohere no configurada, se deshabilitará el reranking.")

if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    log_message("USER_AGENT configurado en las variables de entorno.")

# --------------------- Conexión a Chroma y Creación de Embeddings ---------------------
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
log_message("Embeddings creados con OpenAI.")

# Nota: Se reutiliza el vector store existente, aunque ahora se usará solo para la parte semántica (Chroma)
vector_store = Chroma(
    collection_name=collection_name_fragmento,
    persist_directory=fragment_store_directory,
    embedding_function=embeddings
)
log_message("Vector store cargado correctamente desde Chroma.")

# --------------------- Funciones Auxiliares de Texto ---------------------
def count_words(text):
    """Cuenta el número de palabras en un texto."""
    return len(text.split())

def validar_palabras(prompt, max_palabras=1000000):
    """Valida si el prompt excede el límite de palabras permitido."""
    num_palabras = count_words(prompt)
    if num_palabras > max_palabras:
        log_message(f"El contenido supera el límite de {max_palabras} palabras ({num_palabras} utilizadas).")
        return False, num_palabras
    return True, num_palabras

def reducir_contenido_por_palabras(text, max_palabras=1000000):
    """Recorta el texto para no superar el límite de palabras."""
    palabras = text.split()
    if len(palabras) > max_palabras:
        log_message("El contenido es demasiado largo, truncando...")
        return " ".join(palabras[:max_palabras]) + "\n\n[Contenido truncado...]"
    return text

# --------------------- Nuevas Funciones de Recuperación (Script 2) ---------------------
def clean_query(query):
    """Limpia la consulta eliminando caracteres especiales para FTS5."""
    return re.sub(r'[^\w\s]', '', query)

def retrieve_bm25(query):
    """Búsqueda BM25 con SQLite FTS5."""
    log_message("🔍 Consultando BM25...")
    results = []
    try:
        conn = sqlite3.connect(bm25_db_path)
        cursor = conn.cursor()
        safe_query = clean_query(query)
        cursor.execute(
            "SELECT chunk_content FROM chunks WHERE chunk_content MATCH ? LIMIT ?",
            (safe_query, max_results_bm25)
        )
        results = [{"content": row[0], "source": "BM25"} for row in cursor.fetchall()]
        conn.close()
        log_message(f"BM25 encontró {len(results)} resultados.")
    except Exception as e:
        log_message(f"❌ Error BM25: {str(e)}", level="ERROR")
    return results

def retrieve_chromadb(query):
    """Búsqueda semántica con ChromaDB utilizando el vector store existente."""
    log_message("🔍 Consultando ChromaDB...")
    results = []
    try:
        # Se reutiliza el vector_store creado anteriormente
        docs = vector_store.similarity_search_with_score(query, k=max_results_chroma)
        results = [{
            "content": doc.page_content,
            "score": score,
            "source": "ChromaDB"
        } for doc, score in docs]
        log_message(f"ChromaDB encontró {len(results)} resultados.")
    except Exception as e:
        log_message(f"❌ Error ChromaDB: {str(e)}", level="ERROR")
    return results


def rank_fusion(bm25_results, chroma_results):
    """Fusión híbrida mejorada usando Reciprocal Rank Fusion (RRF)"""
    #print("\n🔄 Fusionando resultados con RRF...")
    combined = {}
    
    # Constante de suavizado para RRF (típicamente entre 60-100)
    RRF_K = 60
    
    # Procesar resultados de ChromaDB con RRF
    for chroma_rank, chroma_res in enumerate(chroma_results, 1):
        key = chroma_res['content'][:150]  # Clave más larga para mejor deduplicación
        rrf_score = 1 / (chroma_rank + RRF_K)
        
        combined[key] = {
            **chroma_res,
            'rrf_score': rrf_score,
            'sources': ['ChromaDB']
        }

    # Procesar resultados BM25 con RRF y combinar
    for bm25_rank, bm25_res in enumerate(bm25_results, 1):
        key = bm25_res['content'][:150]
        rrf_score = 1 / (bm25_rank + RRF_K)
        
        if key in combined:
            combined[key]['rrf_score'] += rrf_score
            combined[key]['sources'].append('BM25')
        else:
            combined[key] = {
                **bm25_res,
                'rrf_score': rrf_score,
                'sources': ['BM25']
            }

    # Ordenar por puntuación RRF combinada
    sorted_results = sorted(
        combined.values(), 
        key=lambda x: x['rrf_score'], 
        reverse=True
    )

    # Normalizar scores para mejor interpretación
    max_score = max(r['rrf_score'] for r in sorted_results) if sorted_results else 1
    for res in sorted_results:
        res['score'] = res['rrf_score'] / max_score  # Normalizado 0-1
        res['source'] = ' + '.join(res['sources'])
        del res['rrf_score']
        del res['sources']

    return sorted_results[:rerank_top_n]

def cohere_rerank(query, documents):
    """Reorganización contextual con Cohere."""
    log_message("🎯 Reranking con Cohere...")
    if not co:
        log_message("⚠️ Cohere no configurado. Saltando reranking.")
        return documents
    try:
        texts = [doc['content'] for doc in documents]
        response = co.rerank(
            query=query,
            documents=texts,
            top_n=rerank_top_k,
            model='rerank-multilingual-v2.0'
        )
        # Reordenar documentos según los índices devueltos por Cohere
        reranked = [documents[r.index] for r in response.results]
        log_message("Reranking completado con Cohere.")
        return reranked
    except Exception as e:
        log_message(f"❌ Error Cohere: {str(e)}", level="ERROR")
        return documents[:rerank_top_k]


def retrieve(query):
    """
    Recupera documentos relevantes utilizando una combinación de BM25 y búsqueda semántica en ChromaDB.
    
    Parámetros:
    query (str): Consulta ingresada por el usuario.

    Retorna:
    list: Lista de documentos ordenados según relevancia.
    """
    # Búsquedas paralelas
    bm25_res = retrieve_bm25(query)
    chroma_res = retrieve_chromadb(query)

    # Fusión híbrida mejorada
    fused = rank_fusion(bm25_res, chroma_res)

    
    log_message(f"FUSED...........> {fused}\nFIN FUSED \n\n ------------.")
   


    # Reranking contextual (se mantiene igual)
    if rerank_enabled and len(fused) > 1:
        fused = cohere_rerank(query, fused)

    # Resultados finales
    
    return fused



# --------------------- Grafo Conversacional (LangGraph) ---------------------
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI  # Se utiliza el modelo de lenguaje configurado

llm = ChatOpenAI(model=model_name, temperature=0)  # Ajusta parámetros según sea necesario

# Nodo 1: Generar consulta o responder directamente
def query_or_respond(state: MessagesState):
    """Genera una consulta para la herramienta de recuperación o responde directamente."""
    log_message("########### QUERY OR RESPOND ---------#####################")
    #llm_with_tools = llm.bind_tools([retrieve])
    llm_with_tools = llm.bind_tools(
        [retrieve], 
        tool_choice={"type": "function", "function": {"name": "retrieve"}}  # Forzar llamada
    )
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Nodo 2: Ejecutar la herramienta de recuperación
tools = ToolNode([retrieve])

# Nodo 3: Generar la respuesta final
def generate(state: MessagesState):
    """Genera la respuesta final usando los documentos recuperados."""
    log_message("########### WEB-generate ---------#####################")
    recent_tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"]
    
    # 1. Extraer y formatear documentos
    docs_content = "\n\n".join([
        f"📄 DOCUMENTO {i+1}:\n{msg.content}\n" 
        for i, msg in enumerate(recent_tool_messages[::-1])
    ])
    
    log_message(f"DOCUMENTOS RECUPERADOS:\n{docs_content}")
    # Validar si los documentos contienen términos clave de la pregunta
    user_question = state["messages"][0].content.lower()
    terms = user_question.split()
    
    if not any(term in docs_content.lower() for term in terms):
        return {"messages": [{"role": "assistant", "content": "Lo siento, no tengo información suficiente para responder esa pregunta."}]}
    
    # Construcción del mensaje del sistema con contexto y ejemplo (se mantiene el formato original)
     # Construcción del mensaje del sistema con contexto y ejemplo (se mantiene el formato original)
    system_message_content = ( """
<CONTEXTO>
La información proporcionada tiene como objetivo apoyar a los agentes que trabajan en las agencias de PAMI, quienes se encargan de atender las consultas de los afiliados. Este soporte está diseñado para optimizar la experiencia de atención al público y garantizar que los afiliados reciban información confiable y relevante en el menor tiempo posible.
</CONTEXTO>

<ROL>
   Eres un asistente virtual experto en los servicios y trámites de PAMI.
</ROL>
<TAREA>
   Tu tarea es responder preguntas relacionadas con lo trámites y servicios que ofrece la obra social PAMI, basándote únicamente en los datos disponibles en la base de datos vectorial. Si la información no está disponible, debes decir 'No tengo esa información en este momento'.
</TAREA>

<MODO_RESPUESTA>
<EXPLICACIÓN>
En tu respuesta debes:
-Ser breve y directa: Proporciona la información en un formato claro y conciso, enfocándote en los pasos esenciales o la acción principal que debe tomarse.
-Ser accionable: Prioriza el detalle suficiente para que el agente pueda transmitir la solución al afiliado rápidamente o profundizar si es necesario.
-Evitar información innecesaria: Incluye solo los datos más relevantes para resolver la consulta. Si hay pasos opcionales o detalles adicionales, indícalos solo si son críticos.
-Estructura breve: Usa puntos clave, numeración o listas de una sola línea si es necesario.
-El contenido de la respuesta debe estar orientado a lo que debe hacer el Afiliado 
</EXPLICACION> 

   <EJEMPLO_MODO_RESPUESTA>
      <PREGUNTA>
         ¿Cómo tramitar la insulina tipo glargina?
      </PREGUNTA>
      <RESPUESTA>
         PAMI cubre al 100% la insulina tipo Glargina para casos especiales, previa autorización por vía de excepción. Para gestionarla, se debe presentar el Formulario de Insulinas por Vía de Excepción (INICIO o RENOVACIÓN) firmado por el médico especialista, acompañado de los últimos dos análisis de sangre completos (hemoglobina glicosilada y glucemia, firmados por un bioquímico), DNI, credencial de afiliación y receta electrónica. La solicitud se presenta en la UGL o agencia de PAMI y será evaluada por Nivel Central en un plazo de 72 horas. La autorización tiene una vigencia de 12 meses.
         
      </RESPUESTA>
   </EJEMPLO_MODO_RESPUESTA>
</MODO_RESPUESTA>

<CASOS_DE_PREGUNTA_RESPUESTA>
        <REQUISITOS>
        Si la respuesta tiene requisitos listar **TODOS** los requisitos encontrados en el contexto no omitas      incluso si aparecen en chunks distintos o al final de un fragmento. 
**Ejemplo crítico**: Si un chunk menciona "DNI, recibo, credencial" y otro agrega "Boleta de luz ", DEBEN incluirse ambos.
                             
         **Advertencia**:
          Si faltan requisitos del contexto en tu respuesta, se considerará ERROR GRAVE.                         
        </REQUISITOS>
       
   <IMPORTANTES_Y_EXCEPCIONES>
      Si los servicios o trámites tienen EXCEPCIONES, aclaraciones o detalles IMPORTANTES, EXCLUSIONES, menciónalos en tu respuesta.
        <EJEMPLO>
           ### Exclusiones:
            Afiliados internados en geriaticos privados
           ### Importante
            La orden tiene un vencimiento de 90 dias
           ### Excepciones
            Las solicitudes por vulnerabilidad no tendrán vencimiento
        </EJEMPLO>                      
   </IMPORTANTES_Y_EXCEPCIONES>

   <TRAMITES_NO_DISPONIBLES>
      <EXPLICACION>
         Si la pregunta es sobre un trámite o servicio que no está explícitamente indicado en la base de datos vectorial, menciona que no existe ese trámite o servicio.
      </EXPLICACION>
      <EJEMPLO>
         <PREGUNTA>
            ¿Cómo puede un afiliado solicitar un descuento por anteojos?
         </PREGUNTA>
         <RESPUESTA>
            PAMI no brinda un descuento por anteojos,Por lo tanto, si el afiliado decide comprar los anteojos por fuera de la red de ópticas de PAMI, no será posible solicitar un reintegro.
         </RESPUESTA>
      </EJEMPLO>
   </TRAMITES_NO_DISPONIBLES>

   <CALCULOS_NUMERICOS>
      <EXPLICACION>
         Si la pregunta involucra un cálculo o comparación numérica, evalúa aritméticamente para responderla.
      </EXPLICACION>
      <EJEMPLO>
         - Si se dice "menor a 10", es un número entre 1 y 9.
         - Si se dice "23", es un número entre 21 y 24.
      </EJEMPLO>
   </CALCULOS_NUMERICOS>

   <FORMATO_RESPUESTA>
      <EXPLICACION>
         Presenta la información en formato de lista Markdown si es necesario.
      </EXPLICACION>
   </FORMATO_RESPUESTA>

   <REFERENCIAS>
      <EXPLICACION>
         Al final de tu respuesta, incluye siempre un apartado titulado **Referencias** que contenga combinaciones únicas de **ID_SUB** y **SUBTIPO**, más un link con la siguiente estructura:
      </EXPLICACION>
      <EJEMPLO>
         Referencias:
         - ID_SUB = 347 | SUBTIPO = 'Traslados Programados'
         - LINK = https://simap.pami.org.ar/subtipo_detalle.php?id_sub=347
      </EJEMPLO>
   </REFERENCIAS>
</CASOS_DE_PREGUNTA_RESPUESTA>

"""
    + docs_content 
     )
    
  

    system_message_content = ( """
<CONTEXTO>
La información proporcionada tiene como objetivo apoyar a los agentes que trabajan en las agencias de PAMI, quienes se encargan de atender las consultas de los afiliados. Este soporte está diseñado para optimizar la experiencia de atención al público y garantizar que los afiliados reciban información confiable y relevante en el menor tiempo posible.
</CONTEXTO>

<ROL>
   Eres un asistente virtual experto en los servicios y trámites de PAMI.
</ROL>
<TAREA>
   Tu tarea es responder preguntas relacionadas con lo trámites y servicios que ofrece la obra social PAMI, basándote únicamente en los datos disponibles en la base de datos vectorial. Si la información no está disponible, debes decir 'No tengo esa información en este momento'.
</TAREA>

<MODO_RESPUESTA>
<EXPLICACIÓN>
En tu respuesta debes:
-Ser breve y directa: Proporciona la información en un formato claro y conciso, enfocándote en los pasos esenciales o la acción principal que debe tomarse.
-Ser accionable: Prioriza el detalle suficiente para que el agente pueda transmitir la solución al afiliado rápidamente o profundizar si es necesario.
-Evitar información innecesaria: Incluye solo los datos más relevantes para resolver la consulta. Si hay pasos opcionales o detalles adicionales, indícalos solo si son críticos.
-Estructura breve: Usa puntos clave, numeración o listas de una sola línea si es necesario.
-El contenido de la respuesta debe estar orientado a lo que debe hacer el Afiliado 
                              
-Es importante indicar donde se realiza el tramite si en la Agencia, en la web ,etc
</EXPLICACION> 

   <EJEMPLO_MODO_RESPUESTA>
        <PREGUNTA1>                       
      <PREGUNTA2>
         ¿Cómo tramitar la insulina tipo glargina?
      </PREGUNTA2>
      <RESPUESTA2>
        PAMI cubre al 100% la insulina tipo Glargina para casos especiales, previa autorización por vía de excepción. 
        Para gestionarla el AFILIADO , debe presentar:
1. Formulario de Insulinas por Vía de Excepción (INICIO o RENOVACIÓN) firmado por el médico especialista.
2. Últimos dos análisis de sangre completos (hemoglobina glicosilada y glucemia), firmados por un bioquímico.
....

##### Donde se realiza el trámite
-El trámite se reaiza en forma presencial en la UGL
### Importante
- EL afiliado debe estar registrado previamente en el Padrón de personas con Diabetes.

      </RESPUESTA>
   </EJEMPLO_MODO_RESPUESTA>
</MODO_RESPUESTA>

<CASOS_DE_PREGUNTA_RESPUESTA>
        <REQUISITOS>
        Si la respuesta tiene requisitos listar **TODOS** los requisitos encontrados en el contexto no omitas      incluso si aparecen en chunks distintos o al final de un fragmento. 
**Ejemplo crítico**: Si un chunk menciona "DNI, recibo, credencial" y otro agrega "Boleta de luz ", DEBEN incluirse ambos.
                             
         **Advertencia**:
          Si faltan requisitos del contexto en tu respuesta, se considerará ERROR GRAVE.                         
        </REQUISITOS>
       
   <IMPORTANTES_Y_EXCEPCIONES>
      Si los servicios o trámites tienen EXCEPCIONES, aclaraciones o detalles IMPORTANTES, EXCLUSIONES, menciónalos en tu respuesta.
        <EJEMPLO>
           ### Exclusiones:
            Afiliados internados en geriaticos privados
           ### Importante
            La orden tiene un vencimiento de 90 dias
           ### Excepciones
            Las solicitudes por vulnerabilidad no tendrán vencimiento
        </EJEMPLO>                      
   </IMPORTANTES_Y_EXCEPCIONES>

   <TRAMITES_NO_DISPONIBLES>
      <EXPLICACION>
         Si la pregunta es sobre un trámite o servicio que no está explícitamente indicado en la base de datos vectorial, menciona que no existe ese trámite o servicio.
      </EXPLICACION>
      <EJEMPLO>
         <PREGUNTA>
            ¿Cómo puede un afiliado solicitar un descuento por anteojos?
         </PREGUNTA>
         <RESPUESTA>
            PAMI no brinda un descuento por anteojos,Por lo tanto, si el afiliado decide comprar los anteojos por fuera de la red de ópticas de PAMI, no será posible solicitar un reintegro.
         </RESPUESTA>
      </EJEMPLO>
   </TRAMITES_NO_DISPONIBLES>

   <CALCULOS_NUMERICOS>
      <EXPLICACION>
         Si la pregunta involucra un cálculo o comparación numérica, evalúa aritméticamente para responderla.
      </EXPLICACION>
      <EJEMPLO>
         - Si se dice "menor a 10", es un número entre 1 y 9.
         - Si se dice "23", es un número entre 21 y 24.
      </EJEMPLO>
   </CALCULOS_NUMERICOS>

   <FORMATO_RESPUESTA>
      <EXPLICACION>
         Presenta la información en formato de lista Markdown si es necesario.
      </EXPLICACION>
   </FORMATO_RESPUESTA>

   <REFERENCIAS>
      <EXPLICACION>
         Al final de tu respuesta, incluye siempre un apartado titulado **Referencias** que contenga combinaciones únicas de **ID_SUB** y **SUBTIPO**, más un link con la siguiente estructura:
      </EXPLICACION>
      <EJEMPLO>
         Referencias:
         - ID_SUB = 347 | SUBTIPO = 'Traslados Programados'
         - LINK = https://simap.pami.org.ar/subtipo_detalle.php?id_sub=347
      </EJEMPLO>
   </REFERENCIAS>
</CASOS_DE_PREGUNTA_RESPUESTA>

"""
    + docs_content 
     )
    
    # Validar límite de palabras
    es_valido, num_palabras = validar_palabras(system_message_content)
    if not es_valido:
        system_message_content = reducir_contenido_por_palabras(system_message_content)
        log_message(f"Se redujo el contenido a {count_words(system_message_content)} palabras.")
    
    # Loguear el contexto que se usará en el prompt
    log_message("Contexto de prompt para consulta:\n" + system_message_content)
    
    prompt = [SystemMessage(system_message_content)] + [
        msg for msg in state["messages"] if msg.type in ("human", "system")
    ]

# Filtrar solo el último mensaje humano y cualquier system message relevante
    human_messages = [msg for msg in state["messages"] if msg.type == "human"]
    last_human_message = human_messages[-1] if human_messages else None

    system_messages = [msg for msg in state["messages"] if msg.type == "system"]

    prompt = [
        SystemMessage(content=system_message_content),  # Nuevo sistema con contexto
        last_human_message,  # Última pregunta
        *system_messages  # Mantener system messages existentes si son críticos
    ]

    prompt = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=state["messages"][0].content)  # La pregunta original
    ]
    # Debug: Verificar prompt completo
    
    
    log_message(f"\n\n\nWEB-PROMPTJJJJ: {prompt} \n\n---FIN WEB-PROMPT")
    response = llm.invoke(prompt)
    log_message(f"Respuesta del LLM: {response}")
    return {"messages": [response]}

# Construcción y conexión del grafo
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_edge("query_or_respond", "tools")
graph_builder.add_edge("tools", "generate")
graph = graph_builder.compile()

# --------------------- Función para Procesar Preguntas ---------------------
def process_question(question_input: str, fecha_desde: str, fecha_hasta: str, k: int):
    log_message("############## Iniciando process_question ##############")
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    try:
        for step in graph.stream(
            {"messages": [{"role": "user", "content": question_input}]},
            stream_mode="values",
            config={"configurable": {"thread_id": "user_question"}},
        ):
            response = step["messages"][-1].content
            log_message("############## Fin process_question jaja ##############")
        return response
    except Exception as e:
        log_message(f"Error en process_question: {str(e)}", level="ERROR")
        return f"Error: {str(e)}"

# --------------------------------------------------------------------
# Fin del script mejorado
# --------------------------------------------------------------------

# Custo v-1.0.3
