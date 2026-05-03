import os
import asyncio
import pandas as pd
from opensearchpy import OpenSearch

# Importación de funciones de los módulos
from create_csv.create_csv import ingest_logs_to_csv_stream, split_dataset
from create_csv.downsampling import apply_downsampling
from OpenSearch.opensearch_manager import index_knn
from knn.ingest_knn import ingest_logs
from knn.inference_knn import knn_rag, build_prompt_ids
from LLM.llm_connector import get_llm_ids

# CONFIGURACIÓN DE RUTAS
# Cambia estas rutas a las carpetas donde descargaste el dataset FOX
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GATHER_DIR = os.path.join(BASE_DIR, "raw_dataset", "gather")
# Archivo temporal con todos los logs (muy pesado)
RAW_CSV = os.path.join(BASE_DIR, "logs-dataset-raw.csv")
# Archivo final después del downsampling
OUTPUT_CSV = os.path.join(BASE_DIR, "logs-dataset-reduced.csv")
# Configuración de OpenSearch
INDEX_NAME = "logs-index-tfm"

# CONFIGURACIÓN DE CONEXIÓN OPENSEARCH (DOCKER WINDOWS)
auth = ('admin', 'admin') # Credenciales por defecto
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_compress=True,
    http_auth=auth,
    use_ssl=False, # Cambiar a True si configuras certificados
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

def check_structure():
    """Verifica que las carpetas existan antes de empezar"""
    if not os.path.exists(GATHER_DIR):
        print(f"ERROR: No se encuentra la carpeta de logs en: {GATHER_DIR}")
        print("Crea la carpeta 'gather' y mete ahí los archivos .log")
        return False
    return True

if __name__ == "__main__":
    print("--- Inicio del proceso de prueba ---")
    
    if check_structure():
        # PASO 1: Ingesta de datos (Crear el CSV)
        # Esto recorrerá tus logs y creará el archivo estructurado
        print("Pasando logs a CSV... (esto puede tardar según el volumen)")
        ingest_logs_to_csv_stream(GATHER_DIR, RAW_CSV)
        
        # PASO 2: Downsampling
        print("Aplicando downsampling...")
        apply_downsampling(RAW_CSV, OUTPUT_CSV, n_normal_samples=50000)

        # PASO 3: Configurar OpenSearch
        # Creamos el índice con motor Faiss y algoritmo HNSW según el TFG
        print("\n[3/4] Configurando índice k-NN en OpenSearch...")
        try:
            index_knn(client, INDEX_NAME)
            ingest_logs(client, OUTPUT_CSV, INDEX_NAME)
        except Exception as e:
            print(f"Error conectando a OpenSearch: {e}")
            print("Asegúrate de que el contenedor de Docker esté corriendo.")

        # PASO 4: Visualización rápida
        if os.path.exists(OUTPUT_CSV):
            print("\n--- Vista previa de los datos extraídos ---")
            df = pd.read_csv(OUTPUT_CSV)
            print(df.head(10)) # Muestra las primeras 10 filas
            
            print("\n--- Resumen de etiquetas encontradas ---")
            # Esto te dirá cuántos logs son 'normal_log' y cuántos son ataques
            print(df['label'].value_counts())
            
            print("\n--- Resumen de fuentes (Source) ---")
            print(df['source'].value_counts())

            # PASO 3: Probar el split (Entrenamiento / Evaluación)
            print("\nDividiendo el dataset en Train y Test...")
            split_dataset(OUTPUT_CSV)
        else:
            print("Error: El archivo CSV no se generó.")

    print("\n--- Proceso finalizado ---")