from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# pyrefly: ignore[missing-import]
from langdetect import detect, DetectorFactory
# pyrefly: ignore[missing-import]
from deep_translator import GoogleTranslator

# Fijar semilla para que langdetect sea determinista
DetectorFactory.seed = 0

# Descargar recursos ligeros de NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = FastAPI(title="API Detector de Phishing")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. DETECCIÓN DE IDIOMA Y TRADUCCIÓN LIGERA EN RAM (<512 MB) ---
def detectar_idioma(texto):
    try:
        idioma = detect(texto)
        if idioma == "es":
            return "español"
        elif idioma == "en":
            return "ingles"
        else:
            return "no compatible"
    except Exception:
        return "no compatible"

def traducir_ingles(texto):
    idioma = detectar_idioma(texto)

    if idioma == "español":
        try:
            texto = GoogleTranslator(source='es', target='en').translate(texto[:4500])
        except Exception:
            pass

    return texto

# --- 2. LIMPIEZA DE TEXTO DEL NOTEBOOK ---
stop = (set(stopwords.words('english')) - {'now', 'not', 'no'}) | {
    'html', 'alternative', 'strong', 'nbsp', 'font', 'href',
    'enron', 'ect', 'wrote', 'mailing', 'list',
    'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december'
}
lemma = WordNetLemmatizer()

def preparar_texto(t):
    t = t.lower()
    t = re.sub(r'content-type:.*|mime-version:.*|multipart/\S+|text/(html|plain)|charset=\S+', ' ', t)
    t = re.sub(r'-{2,}\s*original message\s*-{2,}', ' ', t)
    t = re.sub(r'on .{0,80}wrote:', ' ', t)
    t = re.sub(r'posting guide.{0,200}?reproducible code', ' ', t, flags=re.DOTALL)
    t = re.sub(r'^(from|to|cc|bcc|sent|date):.*$', ' ', t, flags=re.MULTILINE)
    t = re.sub(r'^subject:', ' ', t, flags=re.MULTILINE)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'http\S+|www\.\S+', ' tokurl ', t)
    t = re.sub(r'\S+@\S+', ' tokemail ', t)
    t = re.sub(r'[$€£]\s?\d[\d,\.]*', ' tokmoney ', t)
    t = re.sub(r'\d+', ' toknum ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    tokens = [lemma.lemmatize(w) for w in t.split() if w not in stop and len(w) > 2]
    return ' '.join(tokens)

# --- 3. CARGAR LOS 4 ARTEFACTOS JOBLIB ---
tfidf = joblib.load('vectorizer_update.joblib')
selector_chi2 = joblib.load('selector_chi2_update.joblib')
best_chromosome = np.array(joblib.load('best_chromosome_update.joblib'))

if os.path.exists('model_ga_update.joblib'):
    model_ga = joblib.load('model_ga_update.joblib')
else:
    model_ga = joblib.load('model.joblib')

if len(best_chromosome) > 1000:
    best_chromosome = best_chromosome[:1000]

class EmailInput(BaseModel):
    texto: str

# --- 4. FLUJO DE PREDICCIÓN ---
def predecir_correo(texto_recibido):
    # 1. Traducir (si aplica) y limpiar el texto usando tus funciones del notebook
    texto_traducido = traducir_ingles(texto_recibido)
    texto_limpio = preparar_texto(texto_traducido)
    
    # 2. Vectorizar con TF-IDF (Pasa de texto a 20,000 dimensiones)
    features_tfidf = tfidf.transform([texto_limpio])
    
    # 3. Reducir con Chi-cuadrado (Pasa de 20,000 a 1,000 dimensiones)
    features_chi2 = selector_chi2.transform(features_tfidf)
    
    # 4. Filtrar con el Cromosoma del Algoritmo Genético
    features_ga = features_chi2[:, best_chromosome == 1]
    
    # 5. Obtener veredicto y probabilidades de Phishing
    prediccion = model_ga.predict(features_ga)[0]
    probabilidades = model_ga.predict_proba(features_ga)[0]
    
    veredicto = "Phishing" if prediccion == 1 else "Legítimo"
    porcentaje_legitimo = probabilidades[0] * 100
    porcentaje_phishing = probabilidades[1] * 100
    
    return {
        "veredicto": veredicto,
        "probabilidad_legitimo": f"{porcentaje_legitimo:.2f}%",
        "probabilidad_phishing": f"{porcentaje_phishing:.2f}%"
    }

@app.get("/")
def health_check():
    return {"estado": "activo"}

@app.post("/api/analizar")
def analizar_correo(datos: EmailInput):
    if not datos.texto or not datos.texto.strip():
        raise HTTPException(status_code=400, detail="El texto del correo no puede estar vacío.")
    return predecir_correo(datos.texto)