from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import re

app = FastAPI(
    title="API de Detección de Phishing (Optimizado con AG)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Carga segura de artefactos sin forzar el cruce de dimensiones al inicio
vectorizer = joblib.load("vectorizer.joblib")
best_chromosome = np.array(joblib.load("best_chromosome.joblib"), dtype=bool)
model = joblib.load("model_ga.joblib")

feature_names = np.array(vectorizer.get_feature_names_out())
n_genes = len(best_chromosome) # 1000 en tu caso

class EmailInput(BaseModel):
    texto: str

def preprocesar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\.\S+', ' enlaceweb ', texto)
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def obtener_indicadores_semanticos(texto_limpio: str) -> list[str]:
    indicadores = []
    if any(p in texto_limpio for p in ["urgente", "inmediatamente", "suspendida", "24 horas", "bloqueada", "alerta", "urgent", "immediately", "suspended"]):
        indicadores.append("urgencia")
    if any(p in texto_limpio for p in ["verifique", "confirmar", "actualizar", "datos", "contraseña", "credenciales", "cuenta", "verify", "account", "password"]):
        indicadores.append("verificación de cuenta")
    if any(p in texto_limpio for p in ["enlaceweb", "clic", "link", "enlace", "acceda", "ingrese", "click", "login"]):
        indicadores.append("solicitud de acceso a enlace")
    return indicadores if indicadores else ["sin indicadores críticos evidentes"]

@app.get("/")
def health_check():
    return {
        "estado": "activo",
        "dimension_tfidf": len(feature_names),
        "dimension_cromosoma_ag": n_genes,
        "caracteristicas_esperadas_modelo": getattr(model, "n_features_in_", "desconocido")
    }

@app.post("/api/analizar")
def analizar_correo(datos: EmailInput):
    if not datos.texto or not datos.texto.strip():
        raise HTTPException(status_code=400, detail="El texto del correo no puede estar vacío.")

    # Paso A: Limpieza del texto
    texto_limpio = preprocesar_texto(datos.texto)

    # Paso B: Vectorización TF-IDF (Genera 20,000 columnas)
    vector_tfidf = vectorizer.transform([texto_limpio]).toarray()

    # Paso C: Ajustar dimensión al tamaño del cromosoma (1,000) y filtrar con el AG
    vector_recortado = vector_tfidf[:, :n_genes]
    nombres_recortados = feature_names[:n_genes]

    # Verificamos cuántas columnas espera realmente model_ga.joblib
    n_esperadas = getattr(model, "n_features_in_", None)

    if n_esperadas == int(np.sum(best_chromosome)):
        # Caso 1: El modelo fue entrenado solo con los genes en 1 del AG
        vector_final = vector_recortado[:, best_chromosome]
        nombres_finales = nombres_recortados[best_chromosome]
    elif n_esperadas == n_genes:
        # Caso 2: El modelo espera las 1,000 columnas multiplicadas por la máscara (0 y 1)
        vector_final = vector_recortado * best_chromosome
        nombres_finales = nombres_recortados
    else:
        # Caso 3: El modelo espera las 20,000 columnas originales
        vector_final = vector_tfidf
        nombres_finales = feature_names

    # Paso D: Predicción con el modelo
    prediccion_clase = model.predict(vector_final)[0]
    es_phishing = str(prediccion_clase).lower() in ['1', 'phishing', 'spam', 'true', 'posible phishing']
    veredicto = "posible phishing" if es_phishing else "legítimo"

    # Probabilidad
    if hasattr(model, "predict_proba"):
        probabilidades = model.predict_proba(vector_final)[0]
        prob_phishing = round(float(probabilidades[1]) * 100, 2) if len(probabilidades) > 1 else 100.0
    else:
        prob_phishing = 100.0 if es_phishing else 0.0

    # Paso E: Términos detectados
    indices_activos = np.where(vector_final[0] > 0)[0]
    terminos_presentes = nombres_finales[indices_activos].tolist()

    return {
        "texto_original": datos.texto,
        "texto_procesado": texto_limpio,
        "clasificacion": veredicto,
        "probabilidad_phishing": prob_phishing,
        "terminos_detectados_ag": terminos_presentes,
        "indicadores_textuales": obtener_indicadores_semanticos(texto_limpio)
    }