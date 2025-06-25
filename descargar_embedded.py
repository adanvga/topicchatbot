import os
import sys
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

from sentence_transformers import SentenceTransformer

# Descarga y guarda el modelo localmente
modelo = SentenceTransformer('all-MiniLM-L6-v2')
modelo.save('./all-MiniLM-L6-v2')

print("✅ Modelo descargado y guardado en ./all-MiniLM-L6-v2")
