import os
import pandas as pd
from opensearchpy import OpenSearch
from sklearn.model_selection import train_test_split

# Importación de funciones de los módulos
from create_csv.create_csv import ingest_logs_to_csv_stream, split_dataset
from create_csv.downsampling import apply_downsampling
from OpenSearch.opensearch_manager import index_knn
from knn.ingest_knn import ingest_logs
from knn.inference_knn import knn_rag, build_prompt_ids
from metrics.metrics_knn import evaluate_knn_classification, generate_confusion_matrix_plot

# CONFIGURACIÓN DE RUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GATHER_DIR = os.path.join(BASE_DIR, "raw_dataset", "gather")
# Archivo temporal con todos los logs (muy pesado)
RAW_CSV = os.path.join(BASE_DIR, "logs-dataset-raw.csv")
# Archivo final después del downsampling
OUTPUT_CSV = os.path.join(BASE_DIR, "logs-dataset-reduced.csv")
# Configuración de OpenSearch
INDEX_NAME = "logs-index-tfm"

# CONFIGURACIÓN DE CONEXIÓN OPENSEARCH
auth = ('admin', 'admin') # Credenciales por defecto
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_compress=True,
    http_auth=auth,
    use_ssl=False,
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
        # PASO 1: Ingesta y preprocesamiento
        print("Pasando logs a CSV...")
        ingest_logs_to_csv_stream(GATHER_DIR, RAW_CSV)
        
        print("Aplicando downsampling...")
        apply_downsampling(RAW_CSV, OUTPUT_CSV, n_normal_samples=50000)

        # PASO 2: División de datos
        print("Dividiendo dataset (Train para Indexar / Test para Evaluar)...")
        df = pd.read_csv(OUTPUT_CSV)
        # Separamos el 20% para test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_csv_path = os.path.join(BASE_DIR, "train_data.csv")
        train_df.to_csv(train_csv_path, index=False)

        # PASO 3: Configuración e ingesta en OpenSearch
        print("\nConfigurando índice knn y alimentando base de datos vectorial...")
        try:
            index_knn(client, INDEX_NAME)
            ingest_logs(client, train_csv_path, INDEX_NAME)
        except Exception as e:
            print(f"Error conectando a OpenSearch: {e}")

        # PASO 4: Evaluación y visualización
        if os.path.exists(OUTPUT_CSV):
            print("\n--- Evaluando rendimiento del modelo knn ---")
            # Probamos el modelo con el set de test
            y_true, y_pred = evaluate_knn_classification(
                client, test_df, INDEX_NAME, knn_rag, n_samples=200
            )
            
            # Generamos la matriz de confusión
            generate_confusion_matrix_plot(y_true, y_pred, "confusion_matrix_tfm.png")
            
            # Estadísticas finales
            print("\n--- Resumen de etiquetas encontradas ---")
            print(df['label'].value_counts())
            
            print(f"\nTotal Logs Procesados: {len(df)}")
        else:
            print("Error: El archivo CSV no se generó.")
        
        # Limpieza de archivo temporal de entrenamiento
        if os.path.exists(train_csv_path):
            os.remove(train_csv_path)

    print("\n--- Proceso finalizado. ---")