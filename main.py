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

# --- 3. CARGAR ARTEFACTOS JOBLIB ---
# Compartidos por los 4 modelos: TF-IDF (20,000) -> Chi² (1,000)
tfidf = joblib.load('vectorizer_update.joblib')
selector_chi2 = joblib.load('selector_chi2_update.joblib')


def _cargar_mascara(ruta):
    """Carga un cromosoma y se queda solo con los primeros 1000 genes
    (los de selección de características; los 5 últimos del AG con
    regularización codifican lambda y no son columnas)."""
    mascara = np.array(joblib.load(ruta))
    return mascara[:1000]


# Cada modelo tiene su modelo entrenado y (opcionalmente) su máscara de características.
# mascara=None -> usa las 1000 características de Chi² (Regresión Logística base).
CONFIG_MODELOS = [
    {
        "id": "lr",
        "nombre": "Regresión Logística (base, K=1000)",
        "modelo": "modelo_base_lr.joblib",
        "mascara": None,
    },
    {
        "id": "ga",
        "nombre": "Algoritmo Genético + Regresión Logística",
        "modelo": "modelo_final_ga.joblib",
        "mascara": "best_chromosome_update.joblib",
    },
    {
        "id": "l1",
        "nombre": "Algoritmo Genético + Regularización L1",
        "modelo": "modelo_final_ga_l1.joblib",
        "mascara": "chromosome_l1.joblib",
    },
    {
        "id": "l2",
        "nombre": "Algoritmo Genético + Regularización L2",
        "modelo": "modelo_final_ga_l2.joblib",
        "mascara": "chromosome_l2.joblib",
    },
]

MODELOS = []
for cfg in CONFIG_MODELOS:
    if not os.path.exists(cfg["modelo"]):
        print(f"[AVISO] No se encontró {cfg['modelo']}; se omite el modelo '{cfg['id']}'.")
        continue
    mascara = None
    if cfg["mascara"] is not None:
        if not os.path.exists(cfg["mascara"]):
            print(f"[AVISO] No se encontró {cfg['mascara']}; se omite el modelo '{cfg['id']}'.")
            continue
        mascara = _cargar_mascara(cfg["mascara"]) == 1
    MODELOS.append({
        "id": cfg["id"],
        "nombre": cfg["nombre"],
        "modelo": joblib.load(cfg["modelo"]),
        "mascara": mascara,
    })

if not MODELOS:
    raise RuntimeError("No se pudo cargar ningún modelo. Revisa los archivos .joblib.")


class EmailInput(BaseModel):
    texto: str

# --- 4. FLUJO DE PREDICCIÓN ---
def predecir_correo(texto_recibido):
    # 1. Traducir (si aplica) y limpiar el texto (una sola vez para los 4 modelos)
    idioma = detectar_idioma(texto_recibido)
    texto_traducido = traducir_ingles(texto_recibido)
    texto_limpio = preparar_texto(texto_traducido)

    # 2. TF-IDF (20,000 dim) y 3. Chi² (1,000 dim): compartidos
    features_tfidf = tfidf.transform([texto_limpio])
    features_chi2 = selector_chi2.transform(features_tfidf)

    # 4. Cada modelo aplica su propia máscara y predice
    resultados = []
    votos_phishing = 0
    for m in MODELOS:
        features = features_chi2 if m["mascara"] is None else features_chi2[:, m["mascara"]]

        prediccion = int(m["modelo"].predict(features)[0])
        probabilidades = m["modelo"].predict_proba(features)[0]

        es_phishing = prediccion == 1
        votos_phishing += int(es_phishing)

        resultados.append({
            "id": m["id"],
            "nombre": m["nombre"],
            "veredicto": "Phishing" if es_phishing else "Legítimo",
            "probabilidad_legitimo": round(float(probabilidades[0]) * 100, 2),
            "probabilidad_phishing": round(float(probabilidades[1]) * 100, 2),
        })

    total = len(resultados)
    return {
        "idioma_detectado": idioma,
        "resultados": resultados,
        "consenso": {
            "veredicto": "Phishing" if votos_phishing > total / 2 else "Legítimo",
            "votos_phishing": votos_phishing,
            "votos_legitimo": total - votos_phishing,
            "total_modelos": total,
        },
    }


@app.get("/")
def health_check():
    return {"estado": "activo", "modelos_cargados": [m["id"] for m in MODELOS]}


@app.post("/api/analizar")
def analizar_correo(datos: EmailInput):
    if not datos.texto or not datos.texto.strip():
        raise HTTPException(status_code=400, detail="El texto del correo no puede estar vacío.")
    return predecir_correo(datos.texto)