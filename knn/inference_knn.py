import json
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

# Cargamos el modelo (debe ser el mismo usado en la ingesta)
model = SentenceTransformer('all-MiniLM-L6-v2')

def knn_rag(client, log_nuevo, index_name, k=3):
    """
    Paso 1: Convertir log nuevo a vector.
    Paso 2: Buscar los k logs más similares en OpenSearch.
    """
    # Generamos embedding del log que queremos analizar
    vector_query = model.encode(log_nuevo).tolist()

    # Consulta knn
    query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector_query,
                    "k": k
                }
            }
        }
    }

    response = client.search(index=index_name, body=query)
    
    # Extraemos los resultados
    contexto = []
    for hit in response['hits']['hits']:
        contexto.append({
            "log": hit['_source']['log_message'],
            "label": hit['_source']['label'],
            "score": hit['_score']
        })
    return contexto

def build_prompt_ids(log_nuevo, contexto):
    """
    Paso 3: Crear el prompt enriquecido para el LLM.
    """
    prompt = f"""Tú eres un experto en ciberseguridad y detección de intrusiones.
Analiza el siguiente LOG ENTRANTE y determina si es una amenaza basándote en los EJEMPLOS SIMILARES recuperados de nuestra base de datos de ataques conocidos.

LOG ENTRANTE A ANALIZAR:
"{log_nuevo}"

EJEMPLOS SIMILARES DETECTADOS ANTERIORMENTE:
"""
    for i, c in enumerate(contexto):
        prompt += f"\nEjemplo {i+1}: {c['log']} | Etiqueta: {c['label']} (Similitud: {c['score']})"

    prompt += """
Responde únicamente en formato JSON con la siguiente estructura:
{
  "label": "clasificación_final",
  "explicacion": "razonamiento breve",
  "es_amenaza": true/false
}
"""
    return prompt