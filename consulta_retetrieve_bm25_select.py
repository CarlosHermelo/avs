import sqlite3
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
#Ruta de la base de datos
bm25_db_path = config['SERVICIOS_SIMAP_ANTRO'].get('BM25_DB_PATH')
  # Asegúrate de que esta es la ruta correcta

def mostrar_contenido_bm25():
    """Muestra todos los datos almacenados en la base de datos BM25 (FTS5)"""
    try:
        conn = sqlite3.connect(bm25_db_path)
        c = conn.cursor()
        
        # Obtener todas las filas de la tabla chunks
        c.execute("SELECT rowid, chunk_content FROM chunks WHERE chunk_content MATCH 'Afiliacion hijos estudiantes del titular hasta 25 años inclusive' ")  # Muestra hasta 50 registros
        rows = c.fetchall()
        
        conn.close()
        
        if rows:
            print("\n📌 **Contenido actual en BM25 (SQLite FTS5):**")
            for row in rows:
                print(f"\n🔹 ID: {row[0]}")
                print(f"📝 Texto Indexado: {row[1]}")
                print("-" * 80)
        else:
            print("⚠️ No hay datos almacenados en BM25.")

    except Exception as e:
        print(f"❌ Error al acceder a BM25: {str(e)}")

# Ejecutar la inspección
mostrar_contenido_bm25()
