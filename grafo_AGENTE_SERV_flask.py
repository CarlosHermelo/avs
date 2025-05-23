# 🛠 Configuración e Instalación de Dependencias

# 🔑 Configuración de la API Key de OpenAI
import getpass
import os
import configparser
import logging
import tiktoken  # Agregamos tiktoken para contar tokens
import datetime
import json

# Configuración del logging
glog_filename = 'script_log.log'  # Nombre del archivo de log
logging.basicConfig(
    filename=glog_filename,
    filemode='a',  # Agregar registros al final del archivo existente
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Nivel de logging (puedes ajustar según sea necesario)
)

# Crear una función de envoltura para redirigir prints a logging
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

# Cargar la configuración desde config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Leer las variables API Key y modelo desde el archivo de configuración
collection_name_fragmento = config['DEFAULT'].get('collection_name_fragmento', fallback='fragment_store')  # Nombre explícito para la colección desde config.ini
model_name = config['DEFAULT'].get('modelo')
log_message(f"Modelo configurado: {model_name}")

# Obtener la ruta del almacenamiento de fragmentos
fragment_store_directory = config['SERVICIOS_SIMAP'].get('FRAGMENT_STORE_DIR', fallback='/content/chroma_fragment_store')
max_results = config['SERVICIOS_SIMAP'].getint('max_results', fallback=4)
fecha_desde_pagina = config['SERVICIOS_SIMAP'].get('fecha_desde', fallback='2024-01-08')
fecha_hasta_pagina = config['SERVICIOS_SIMAP'].get('fecha_hasta', fallback='2024-12-10')

log_message(f"Fragment store directory: {fragment_store_directory}")
log_message(f"Max results configurado: {max_results}")
log_message(f"Rango de fechas configurado: {fecha_desde_pagina} a {fecha_hasta_pagina}")

# Establecer la clave API de OpenAI en la variable de entorno
api_key = config['DEFAULT'].get('openai_api_key')
os.environ['OPENAI_API_KEY'] = api_key
log_message("API Key configurada.")

# Establecer la variable de entorno USER_AGENT
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    log_message("USER_AGENT configurado en las variables de entorno.")

# Cargar documentos y crear la conexión al vector store en Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Crear embeddings
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
log_message("Embeddings creados con OpenAI.")

# Conectar al vector store existente en Chroma
vector_store = Chroma(
    collection_name=collection_name_fragmento,
    persist_directory=fragment_store_directory,
    embedding_function=embeddings
)
log_message("Vector store cargado correctamente desde Chroma.")

def count_words(text):
    """Cuenta el número de palabras en un texto."""
    return len(text.split())

def validar_palabras(prompt, max_palabras=10000):
    """Valida si el prompt excede el límite de palabras permitido."""
    num_palabras = count_words(prompt)
    if num_palabras > max_palabras:
        log_message(f"\nEl contenido supera el límite de {max_palabras} palabras ({num_palabras} palabras utilizadas).")
        return False, num_palabras
    return True, num_palabras
def reducir_contenido_por_palabras(text, max_palabras=10000):
    """Recorta el texto para no superar el límite de palabras."""
    palabras = text.split()
    if len(palabras) > max_palabras:
        log_message("El contenido es demasiado largo, truncando...")
        return " ".join(palabras[:max_palabras]) + "\n\n[Contenido truncado...]"
    return text

# 🔧


# 🔧 Crear la Lógica de Recuperación (Retrieve Tool)
def retrieve(query: str):
    """Recuperar información relacionada con la consulta."""
    log_message(f"########### RETRIEVE --------#####################")
    
    # Contamos tokens de la consulta
    tokens_consulta = contar_tokens(query, model_name)
    log_message(f"Tokens de entrada en retrieve (consulta): {tokens_consulta}")
    
    retrieved_docs = vector_store.similarity_search_with_score(query, k=max_results)

    documentos_relevantes = [doc for doc, score in retrieved_docs]
    cantidad_fragmentos = len(documentos_relevantes)
    
    # Guardamos la cantidad de fragmentos como variable global para acceder después
    retrieve.last_fragments_count = cantidad_fragmentos

    if not documentos_relevantes:
        log_message("No se encontró información suficiente para responder la pregunta.")
        return "Lo siento, no tengo información suficiente para responder esa pregunta."

    serialized = "\n\n".join(
        (f"fFRAGMENTO{doc.page_content}\nMETADATA{doc.metadata}") for doc in documentos_relevantes
    )
    
    # Contamos tokens de la respuesta de retrieve
    tokens_respuesta_retrieve = contar_tokens(serialized, model_name)
    log_message(f"Fragmentos recuperados de la base de datos vectorial: {cantidad_fragmentos}")
    log_message(f"Tokens de salida en retrieve (documentos): {tokens_respuesta_retrieve}")
    log_message(f"Total tokens en retrieve: {tokens_consulta + tokens_respuesta_retrieve}")
    
    log_message(f"WEB-RETREIVE----> :\n {serialized} \n----------END-WEB-RETRIEBE <")
    return serialized

# Inicializamos el contador de fragmentos
retrieve.last_fragments_count = 0

# Crear el Gráfico de Conversación con LangGraph
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

# Inicialización del gráfico de mensajes
graph_builder = StateGraph(MessagesState)

# Primero define tu modelo de lenguaje
from langchain_openai import ChatOpenAI  # o el modelo que estés usando

llm = ChatOpenAI(model=model_name, temperature=0) # Ajusta los parámetros según necesites

# Función para contar tokens usando tiktoken
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

# Nodo 1: Generar consulta o responder directamente
def query_or_respond(state: MessagesState):
    """Genera una consulta para la herramienta de recuperación o responde directamente."""
    log_message(f"########### QUERY OR RESPOND ---------#####################")
    
    # Contamos tokens de entrada para query_or_respond
    prompt_text = "\n".join([msg.content for msg in state["messages"]])
    tokens_entrada_qor = contar_tokens(prompt_text, model_name)
    log_message(f"Tokens de entrada en query_or_respond: {tokens_entrada_qor}")
    
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    
    # Contamos tokens de salida
    tokens_salida_qor = contar_tokens(response.content, model_name)
    log_message(f"Tokens de salida en query_or_respond: {tokens_salida_qor}")
    log_message(f"Total tokens en query_or_respond: {tokens_entrada_qor + tokens_salida_qor}")
    
    return {"messages": [response]}

# Nodo 2: Ejecutar la herramienta de recuperación
tools = ToolNode([retrieve])

# Nodo 3: Generar la respuesta final
def generate(state: MessagesState):
    """Genera la respuesta final usando los documentos recuperados."""
    log_message(f"###########WEB-generate---------#####################")
    recent_tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"]
    docs_content = "\n\n".join(doc.content for doc in recent_tool_messages[::-1])

    # Validar si los documentos contienen términos clave de la pregunta
    user_question = state["messages"][0].content.lower()

    terms = user_question.split()

    if not any(term in docs_content.lower() for term in terms):
        return {"messages": [{"role": "assistant", "content": "Lo siento, no tengo información suficiente para responder esa pregunta."}]}
         

    
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
Ser breve y directa: Proporciona la información en un formato claro y conciso, enfocándote en los pasos esenciales o la acción principal que debe tomarse.
Ser accionable: Prioriza el detalle suficiente para que el agente pueda transmitir la solución al afiliado rápidamente o profundizar si es necesario.
Evitar información innecesaria: Incluye solo los datos más relevantes para resolver la consulta. Si hay pasos opcionales o detalles adicionales, indícalos solo si son críticos.
Estructura breve: Usa puntos clave, numeración o listas de una sola línea si es necesario.

. </EXPLICACION> 
</MODO_RESPUESTA>

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
   <IMPORTANTES_Y_EXCEPCIONES>
      Si los servicios o trámites tienen excepciones, aclaraciones o detalles IMPORTANTES, menciónalos en tu respuesta.
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
Ser breve y directa: Proporciona la información en un formato claro y conciso, enfocándote en los pasos esenciales o la acción principal que debe tomarse.
Ser accionable: Prioriza el detalle suficiente para que el agente pueda transmitir la solución al afiliado rápidamente o profundizar si es necesario.
Evitar información innecesaria: Incluye solo los datos más relevantes para resolver la consulta. Si hay pasos opcionales o detalles adicionales, indícalos solo si son críticos.
Estructura breve: Usa puntos clave, numeración o listas de una sola línea si es necesario.

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
    

# Validar si excede el límite de palabras
    es_valido, num_palabras = validar_palabras(system_message_content)
    if not es_valido:
        # Reducir el contenido si es necesario
        system_message_content = reducir_contenido_por_palabras(system_message_content)
        log_message(f"###########WEB-Se ha reducido el contenido a {count_words(system_message_content)} palabras.")
        log_message(f"#########WEB-CONTEXTO_QUEDO RESUMIDO ASI (system_message_content\n): {system_message_content} ")

    prompt = [SystemMessage(system_message_content)] + [
        msg for msg in state["messages"] if msg.type in ("human", "system")
    ]
    
    # Contamos tokens del prompt de entrada
    prompt_text = system_message_content + "\n" + "\n".join([msg.content for msg in state["messages"] if msg.type in ("human", "system")])
    tokens_entrada = contar_tokens(prompt_text, model_name)
    log_message(f"Tokens de entrada (prompt): {tokens_entrada}")
    
    log_message(f"WEB-PROMPT  (RESU O NO) System_message_content ------>\n {system_message_content}--<")
    log_message(f"WEB-PROMPT PROMPT ------>\n {prompt}--<")
    
    # Realizamos la inferencia
    response = llm.invoke(prompt)
    
    # Contamos tokens de la respuesta
    tokens_salida = contar_tokens(response.content, model_name)
    log_message(f"Tokens de entrada (respuesta) DE PREGUNTA:: {tokens_entrada}")
    log_message(f"Tokens de salida (respuesta) DE PREGUNTA:: {tokens_salida}")
    log_message(f"Total tokens consumidos DE PREGUNTA: {tokens_entrada + tokens_salida}")
    
    # Añadimos un resumen claro del conteo de tokens
    log_token_summary(tokens_entrada, tokens_salida, model_name)
    
    log_message(f"WEB-PROMPT RESPONSE ------>\n {response}--<")
    return {"messages": [response]}

def log_token_summary(tokens_entrada, tokens_salida, modelo):
    """
    Registra un resumen claro del conteo de tokens para cada inferencia.
    
    Args:
        tokens_entrada (int): Número de tokens de la entrada (pregunta + contexto)
        tokens_salida (int): Número de tokens de la respuesta
        modelo (str): Nombre del modelo utilizado
    """
    # Obtenemos la cantidad de fragmentos recuperados
    cantidad_fragmentos = retrieve.last_fragments_count
    
    separador = "=" * 80
    log_message(separador)
    log_message("RESUMEN DE CONTEO DE TOKENS")
    log_message(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Modelo: {modelo}")
    log_message(separador)
    log_message(f"FRAGMENTOS RECUPERADOS DE LA BD VECTORIAL: {cantidad_fragmentos}")
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
        "fragments_count": cantidad_fragmentos,
        "input_tokens": tokens_entrada,
        "output_tokens": tokens_salida,
        "total_tokens": tokens_entrada + tokens_salida,
        "approx_cost_usd": costo_aprox
    }
    
    log_message(f"RESUMEN_JSON: {json.dumps(resumen_json)}")
    log_message(separador)

# Construcción del gráfico de conversación
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_edge("query_or_respond", "tools")
graph_builder.add_edge("tools", "generate")
graph = graph_builder.compile()

# Función para procesar preguntas
def process_question(question_input: str, fecha_desde: str, fecha_hasta: str, k: int):
    log_message(f"##############-------PROCESSS_QUESTION----------#####################")
    # Registramos tokens de la pregunta inicial
    tokens_pregunta = contar_tokens(question_input, model_name)
    log_message(f"Tokens de la pregunta inicial: {tokens_pregunta}")
    
# 💾 Gestionar el Historial de Conversación (Memory)
    from langgraph.checkpoint.memory import MemorySaver

# Configuración del checkpointer de memoria
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

   
   
    try:
        # Iniciamos contadores
        tokens_totales_entrada = tokens_pregunta
        tokens_totales_salida = 0
        
        # Diccionario para registrar tokens por cada nodo
        tokens_por_nodo = {
            "query_or_respond": {"entrada": 0, "salida": 0},
            "retrieve": {"entrada": 0, "salida": 0},
            "generate": {"entrada": 0, "salida": 0}
        }
        
        for step in graph.stream(
            {"messages": [{"role": "user", "content": question_input}]},
            stream_mode="values",
            config={"configurable": {"thread_id": "user_question"}},
        ):
            response = step["messages"][-1].content
            
            # Este análisis es simplificado, pero podríamos mejorar la detección del nodo actual
            # basándonos en alguna característica específica de la respuesta
            if "query_or_respond" in step.get("logs", []):
                tokens_por_nodo["query_or_respond"]["salida"] += contar_tokens(response, model_name)
            elif "retrieve" in step.get("logs", []):
                tokens_por_nodo["retrieve"]["salida"] += contar_tokens(response, model_name)
            elif "generate" in step.get("logs", []):
                tokens_por_nodo["generate"]["salida"] += contar_tokens(response, model_name)
                
            tokens_totales_salida += contar_tokens(response, model_name)
            
        # Al finalizar registramos el resumen de tokens
        log_message(f"Resumen de consumo de tokens - Inferencia completada:")
        log_message(f"Fragmentos recuperados de la BD vectorial: {retrieve.last_fragments_count}")
        log_message(f"Tokens totales de entrada: {tokens_totales_entrada}")
        log_message(f"Tokens totales de salida: {tokens_totales_salida}")
        log_message(f"Total general de tokens: {tokens_totales_entrada + tokens_totales_salida}")
        
        log_message(f"##############-------FIN ROCESSS_QUESTION----------#####################")
        return response
    except Exception as e:
       
        return f"Error: {str(e)}"