import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# Importación de funciones de los módulos
from create_csv.create_csv import ingest_logs_to_csv_stream
from create_csv.downsampling import apply_downsampling
from metrics.metrics_knn_local import evaluate_knn_local, generate_confusion_matrix_plot

# CONFIGURACIÓN DE RUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GATHER_DIR = os.path.join(BASE_DIR, "raw_dataset", "gather")
RAW_CSV = os.path.join(BASE_DIR, "logs-dataset-raw.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "logs-dataset-reduced.csv")

def check_structure():
    if not os.path.exists(GATHER_DIR):
        print(f"ERROR: No se encuentra la carpeta de logs en: {GATHER_DIR}")
        return False
    return True

if __name__ == "__main__":
    print("--- Inicio del Proyecto TFM (Análisis k-NN Local) ---")
    
    if check_structure():
        # PASO 1: Procesamiento de archivos y creación de CSV
        print("1. Ingestando logs a CSV...")
        ingest_logs_to_csv_stream(GATHER_DIR, RAW_CSV)
        
        print("2. Aplicando downsampling (10k muestras normales para rapidez)...")
        apply_downsampling(RAW_CSV, OUTPUT_CSV, n_normal_samples=10000)

        # PASO 2: División de datos
        print("3. Dividiendo dataset en Entrenamiento y Test...")
        df = pd.read_csv(OUTPUT_CSV)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # PASO 3: Generación de embeddings
        print("4. Generando vectores con SentenceTransformer (all-MiniLM-L6-v2)...")
        model_st = SentenceTransformer('all-MiniLM-L6-v2')
        # Convertimos los logs de entrenamiento en vectores numéricos
        train_embeddings = model_st.encode(train_df['log_message'].tolist())

        # PASO 4: Inicialización del knn
        print("5. Inicializando buscador knn local...")
        # Usamos metric='cosine' para replicar el comportamiento de OpenSearch
        knn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        knn_model.fit(train_embeddings)

        # PASO 5: Evaluación y gráficas
        # Extraemos las etiquetas de entrenamiento para poder predecir
        train_labels = train_df['label'].values
        
        y_true, y_pred = evaluate_knn_local(
            knn_model, train_labels, test_df, model_st, n_samples=len(test_df)
        )
        
        generate_confusion_matrix_plot(y_true, y_pred, "matriz_confusion_tfm.png")

        print("\n--- Estadísticas Finales ---")
        print(df['label'].value_counts())

    print("\n--- Proceso finalizado. Todo se ha ejecutado de forma local ---")