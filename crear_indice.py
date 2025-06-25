# crear_indice.py
import os
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# --- Configuración ---
model = SentenceTransformer('distiluse-base-multilingual-v2')
dimension = 512
index = faiss.IndexFlatL2(dimension)
index_to_doc = {}

def extraer_texto_pdf(path):
    reader = PdfReader(path)
    return "\n".join([page.extract_text() or "" for page in reader.pages]).strip()

def extraer_texto_html(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator=" ").strip()

def leer_documentos(ruta_base, extensiones=('.pdf', '.html')):
    docs = []
    for root, _, files in os.walk(ruta_base):
        for name in files:
            if name.lower().endswith(extensiones):
                ruta = os.path.join(root, name)
                try:
                    texto = extraer_texto_pdf(ruta) if ruta.endswith('.pdf') else extraer_texto_html(ruta)
                    if texto:
                        docs.append(texto)
                except Exception as e:
                    print(f"Error leyendo {ruta}: {e}")
    return docs

def fragmentar(texto, max_palabras=100):
    palabras = texto.split()
    return [" ".join(palabras[i:i+max_palabras]) for i in range(0, len(palabras), max_palabras)]

def construir_y_guardar_indice(ruta_docs, salida_faiss="index.faiss", salida_dict="index_to_doc.pkl"):
    docs = leer_documentos(ruta_docs)
    idx = 0
    total = sum(len(fragmentar(doc)) for doc in docs)

    with tqdm(total=total, desc="Construyendo índice FAISS", ncols=100) as pbar:
        for doc in docs:
            for frag in fragmentar(doc):
                emb = model.encode(frag)
                index.add(np.array([emb], dtype='float32'))
                index_to_doc[idx] = frag
                idx += 1
                pbar.update(1)

    faiss.write_index(index, salida_faiss)
    with open(salida_dict, "wb") as f:
        pickle.dump(index_to_doc, f)

    print(f"✅ Índice guardado en {salida_faiss}")
    print(f"✅ Diccionario guardado en {salida_dict}")

# --- Uso ---
if __name__ == "__main__":
    ruta = "ruta/a/tu/carpeta"
    construir_y_guardar_indice(ruta)
