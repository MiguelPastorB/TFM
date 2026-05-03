from opensearchpy import OpenSearch

def index_knn(client, index_name="logs-index"):
    """
    Configura el índice en OpenSearch con la configuración técnica del TFG:
    Engine: Faiss, Algoritmo: HNSW, Métrica: Cosine Similarity.
    """
    settings = {
        "settings": {
            "index": {
                "knn": True,  # Habilita la funcionalidad knn
                "knn.algo_param.ef_search": "64" # Parámetro de precisión en búsqueda
            }
        },
        "mappings": {
            "properties": {
                # El campo 'embedding' guardará los vectores del Transformer
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,  # Dimensión de all-MiniLM-L6-v2
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil", # Similitud de Coseno
                        "engine": "faiss", # Motor de alto rendimiento
                        "parameters": {
                            "m": 16, # Conexiones por nodo en el grafo
                            "ef_construction": 128
                        }
                    }
                },
                "log_message": {"type": "text"},
                "source": {"type": "keyword"},
                "label": {"type": "keyword"}
            }
        }
    }

    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=settings)
        print(f"Índice '{index_name}' creado exitosamente.")
    else:
        print(f"El índice '{index_name}' ya existe.")