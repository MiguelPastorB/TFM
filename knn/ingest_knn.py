import pandas as pd
from sentence_transformers import SentenceTransformer
from opensearchpy.helpers import bulk

def ingest_logs(client, csv_path, index_name, batch_size=100):
    """
    Carga el modelo de red neuronal, genera embeddings y los sube a OpenSearch.
    """
    print("Cargando modelo Transformer (all-MiniLM-L6-v2)...")
    # Este modelo convierte texto en vectores de 384 dimensiones
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Leyendo datos de {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Procesamos por lotes para mayor eficiencia
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        
        # Generamos los vectores (embeddings)
        # Convertimos la columna log_message en una lista de vectores
        embeddings = model.encode(batch['log_message'].tolist())
        
        acciones = []
        for j, (_, row) in enumerate(batch.iterrows()):
            # Estructura del documento para OpenSearch knn
            doc = {
                "_index": index_name,
                "_source": {
                    "log_message": row['log_message'],
                    "source": row['source'],
                    "label": row['label'],
                    "embedding": embeddings[j].tolist() # El vector generado
                }
            }
            acciones.append(doc)
        
        # Subida masiva a OpenSearch
        bulk(client, acciones)
        print(f"Indexados {i + len(batch)} / {len(df)} logs...")

    print("Indexación vectorial completada.")