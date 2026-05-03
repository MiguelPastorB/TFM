import httpx
import asyncio
import json

# Configuración del LLM (siguiendo los parámetros del TFG)
LLM_API_URL = "http://localhost:11434/v1/chat/completions" # Ejemplo con Ollama
MODEL_NAME = "llama3.1:8b"

async def get_llm_ids(prompt):
    """
    Envía el prompt enriquecido al LLM y recupera la clasificación explicada.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Eres un IDS avanzado. Responde siempre en JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2, # Baja temperatura para mayor determinismo
        "response_format": { "type": "json_object" } # Forzamos salida estructurada
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(LLM_API_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            
            resultado = response.json()
            # Extraemos el contenido del JSON devuelto por el modelo
            content = resultado['choices'][0]['message']['content']
            return json.loads(content)
        
        except Exception as e:
            return {"error": str(e), "label": "error", "explicacion": "Fallo en la conexión con el LLM"}

# Ejemplo de integración rápida:
# async def test():
#    res = await get_llm_ids(mi_prompt_generado)
#    print(f"Predicción: {res['label']} | Razón: {res['explicacion']}")