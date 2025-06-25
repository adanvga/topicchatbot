# consultar_indice.py
import faiss
import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# --- Configuración ---
model = SentenceTransformer('distiluse-base-multilingual-v2')
index = faiss.read_index("index.faiss")

with open("index_to_doc.pkl", "rb") as f:
    index_to_doc = pickle.load(f)

def buscar_contexto(query, k=3):
    emb = model.encode(query)
    D, I = index.search(np.array([emb], dtype='float32'), k)
    return [index_to_doc[i] for i in I[0]]

def preguntar_llm_ollama(query, contexto):
    prompt = f"""<context>
{contexto}
</context>

Pregunta: {query}
Respuesta:"""
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    return r.json()["response"]

# --- Ejemplo de uso ---
if __name__ == "__main__":
    pregunta = input("❓ Pregunta: ")
    contexto = "\n".join(buscar_contexto(pregunta, k=3))
    respuesta = preguntar_llm_ollama(pregunta, contexto)
    print("\n🧠 Respuesta:\n", respuesta)
