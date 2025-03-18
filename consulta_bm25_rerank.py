import sqlite3
import os
import re
import configparser
import numpy as np
import cohere
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder  # <-- Importación del CrossEncoder

# Cargar configuración
config = configparser.ConfigParser()
config.read('config.ini')

# Configuración de APIs
openai_api_key = config['DEFAULT'].get('openai_api_key', '').strip()
cohere_api_key = config['DEFAULT'].get('cohere_api_key', '').strip()
co = cohere.Client(cohere_api_key) if cohere_api_key else None

# Configuración de bases de datos
bm25_db_path = config['SERVICIOS_SIMAP_ANTRO'].get('BM25_DB_PATH', 'bm25_index.db')
fragment_store_directory = config['SERVICIOS_SIMAP_ANTRO'].get('FRAGMENT_STORE_DIR', 'chroma_fragment_store')
collection_name_fragmento = config['DEFAULT'].get('collection_name_fragmento', 'fragment_store')

# Parámetros de recuperación
max_results_bm25 = config['SERVICIOS_SIMAP_ANTRO'].getint('max_results_bm25', 100)
max_results_chroma = config['SERVICIOS_SIMAP_ANTRO'].getint('max_results_chroma', 50)
rerank_enabled = config['SERVICIOS_SIMAP_ANTRO'].getboolean('rerank_enabled', False)
rerank_top_n = config['SERVICIOS_SIMAP_ANTRO'].getint('rerank_top_n', 150)
rerank_top_k = config['SERVICIOS_SIMAP_ANTRO'].getint('rerank_top_k', 20)

# Inicializar el modelo de CrossEncoder de Hugging Face
reranker = CrossEncoder("BAAI/bge-reranker-large")

def clean_query(query):
    """Limpia la consulta para FTS5 eliminando caracteres especiales"""
    return re.sub(r'[^\w\s]', '', query)

def retrieve_bm25(query):
    """Búsqueda BM25 con SQLite FTS5"""
    print("\n🔍 Consultando BM25...")
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
        
    except Exception as e:
        print(f"❌ Error BM25: {str(e)}")
    print(f"BM25---------------------------")
    print(f"Resultado de BM25 \n{results}\n---------\n")
    return results

def retrieve_chromadb(query):
    """Búsqueda semántica con ChromaDB"""
    print("\n🔍 Consultando ChromaDB...")
    results = []
    try:
        if not openai_api_key.startswith('sk-'):
            raise ValueError("API Key OpenAI inválida")
        
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        chroma = Chroma(
            collection_name=collection_name_fragmento,
            persist_directory=fragment_store_directory,
            embedding_function=embeddings
        )
        
        docs = chroma.similarity_search_with_score(query, k=max_results_chroma)
        results = [{
            "content": doc.page_content,
            "score": score,
            "source": "ChromaDB"
        } for doc, score in docs]
        
    except Exception as e:
        print(f"❌ Error ChromaDB: {str(e)}")
    print(f"CHROMA---------------------------")
    print(f" \n\n{results}\n-------FIN---CHROMA \n")
    return results

def rank_fusion(bm25_results, chroma_results):
    """Fusión híbrida mejorada usando Reciprocal Rank Fusion (RRF)"""
    print("\n🔄 Fusionando resultados con RRF...")
    combined = {}
    
    # Constante de suavizado para RRF (típicamente entre 60-100)
    RRF_K = 60
    
    # Procesar resultados de ChromaDB con RRF
    for chroma_rank, chroma_res in enumerate(chroma_results, 1):
        key = chroma_res['content'][:150]  # Clave para deduplicación
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

    # Ordenar por la puntuación RRF combinada
    sorted_results = sorted(
        combined.values(), 
        key=lambda x: x['rrf_score'], 
        reverse=True
    )

    # Normalizar los scores para una mejor interpretación
    max_score = max(r['rrf_score'] for r in sorted_results) if sorted_results else 1
    for res in sorted_results:
        res['score'] = res['rrf_score'] / max_score  # Normalización 0-1
        res['source'] = ' + '.join(res['sources'])
        del res['rrf_score']
        del res['sources']

    return sorted_results[:rerank_top_n]

def cohere_rerank(query, documents):
    """Reorganización contextual con modelo pre-entrenado de Hugging Face usando CrossEncoder"""
    print("\n🎯 Reranking con HuggingFace CrossEncoder...")
    try:
        # Crear pares (consulta, contenido del documento) para evaluar la relevancia
        pairs = [(query, doc['content']) for doc in documents]
        # Obtener puntuaciones de relevancia
        scores = reranker.predict(pairs)
        # Combinar puntuaciones con los documentos
        scored_docs = list(zip(scores, documents))
        # Ordenar los documentos por puntuación (de mayor a menor)
        scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        # Seleccionar los top K documentos reordenados
        reranked_documents = [doc for _, doc in scored_docs[:rerank_top_k]]
        return reranked_documents
    except Exception as e:
        print(f"❌ Error en el reranking: {str(e)}")
        return documents[:rerank_top_k]

def retrieve(query):
    print(f"\n🚀 Iniciando búsqueda para: '{query}'")
    print(f"\n🚀 Valores max_results_chroma: {max_results_chroma}  max_results_bm25: {max_results_bm25} para '{query}'")
    # Búsquedas paralelas
    bm25_res = retrieve_bm25(query)
    chroma_res = retrieve_chromadb(query)

    # Fusión híbrrerank_enableida mejorada
    fused = rank_fusion(bm25_res, chroma_res)

    # Reranking contextual
    if rerank_enabled and len(fused) > 1:
        fused = cohere_rerank(query, fused)

    # Resultados finales
    print(f"\n✅ Total resultados finales: {len(fused)}")
    for idx, res in enumerate(fused[:100], 1):
        print(f"\n🏅 Resultado {idx} ({res['source']}):")
        print(f"📄 {res}")
        print(f"⚖️ Score: {res.get('score', 0):.2f}")
    
    return fused

if __name__ == "__main__":
    # Prueba de integración
    test_query = "¿Usted cobra una pensión por ser Veterano de Guerra?"
    test_query = "¿que cobertura tiene EL SEPELIO del titular?"
    test_query = "¿SI EL titular esta en un geriatrico puede retirar pañales?"

    results = retrieve(test_query)
