from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import re

# 1. Inicialización de la aplicación FastAPI
app = FastAPI(
    title="API de Detección de Phishing (Optimizado con AG)",
    description="Servicio backend para evaluar correos electrónicos desde Angular",
    version="1.0.0"
)

# 2. Configuración de CORS (Indispensable para conectar Vercel con Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción puedes poner el dominio de Vercel: ["https://tu-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Carga de artefactos generados en el Notebook (.ipynb) al iniciar el servidor
try:
    vectorizer = joblib.load("vectorizer.joblib")
    best_chromosome = joblib.load("best_chromosome.joblib")  # Máscara de 0s y 1s del AG
    model = joblib.load("model_ga.joblib")
    feature_names = np.array(vectorizer.get_feature_names_out())
    selected_features_names = feature_names[np.array(best_chromosome, dtype=bool)]
except Exception as e:
    print(f"Advertencia: Asegúrate de colocar los archivos .joblib en la raíz. Error: {e}")


# 4. Modelos de datos (Esquemas de Entrada y Salida con Pydantic)
class EmailInput(BaseModel):
    texto: str


class PredictionOutput(BaseModel):
    texto_original: str
    texto_procesado: str
    clasificacion: str
    probabilidad_phishing: float
    terminos_detectados_ag: list[str]
    indicadores_textuales: list[str]


# 5. Función de preprocesamiento (Debe replicar la limpieza de tu Notebook)
def preprocesar_texto(texto: str) -> str:
    texto = texto.lower()
    # Reemplazar URLs por un token estándar o conservarlas según tu notebook
    texto = re.sub(r'http\S+|www\.\S+', ' enlaceweb ', texto)
    # Eliminar caracteres especiales y puntuación
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)
    # Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


# Función auxiliar para detectar categorías de indicadores según requerimientos
def obtener_indicadores_semanticos(texto_limpio: str) -> list[str]:
    indicadores = []
    if any(p in texto_limpio for p in ["urgente", "inmediatamente", "suspendida", "24 horas", "bloqueada", "alerta"]):
        indicadores.append(" sentido de urgencia o amenaza")
    if any(p in texto_limpio for p in
           ["verifique", "confirmar", "actualizar", "datos", "contraseña", "credenciales", "cuenta"]):
        indicadores.append("solicitud de verificación de cuenta o datos")
    if any(p in texto_limpio for p in ["enlaceweb", "clic", "link", "enlace", "acceda", "ingrese"]):
        indicadores.append("solicitud de acceso a enlace externo")
    return indicadores if indicadores else ["sin indicadores críticos evidentes"]


# 6. Endpoints de la API
@app.get("/")
def health_check():
    return {"estado": "activo", "servicio": "Detector de Phishing API"}


@app.post("/api/analizar", response_model=PredictionOutput)
def analizar_correo(datos: EmailInput):
    if not datos.texto or not datos.texto.strip():
        raise HTTPException(status_code=400, detail="El texto del correo no puede estar vacío.")

    # Paso A: Preparación del texto
    texto_limpio = preprocesar_texto(datos.texto)

    # Paso B: Representación TF-IDF completa
    vector_tfidf = vectorizer.transform([texto_limpio]).toarray()

    # Paso C: Filtrado de características mediante el mejor individuo del Algoritmo Genético
    mascara_booleana = np.array(best_chromosome, dtype=bool)
    vector_filtrado = vector_tfidf[:, mascara_booleana]

    # Paso D: Predicción con el modelo optimizado
    prediccion_clase = model.predict(vector_filtrado)[0]
    probabilidades = model.predict_proba(vector_filtrado)[0]

    # Asumiendo que 1 = phishing y 0 = legítimo
    es_phishing = bool(prediccion_clase == 1)
    prob_phishing = round(float(probabilidades[1]) * 100, 2)
    veredicto = "posible phishing" if es_phishing else "legítimo"

    # Paso E: Identificar qué palabras del AG aparecieron en este correo específico
    indices_activos = np.where(vector_filtrado[0] > 0)[0]
    terminos_presentes = selected_features_names[indices_activos].tolist()

    # Paso F: Construir respuesta JSON
    return PredictionOutput(
        texto_original=datos.texto,
        texto_procesado=texto_limpio,
        clasificacion=veredicto,
        probabilidad_phishing=prob_phishing,
        terminos_detectados_ag=terminos_presentes,
        indicadores_textuales=obtener_indicadores_semanticos(texto_limpio)
    )
