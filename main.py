import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from data.dataset  import load_wav
from src.processor import extraer_features

MODEL_PATH = "model/modelo.pkl"

def predecir(path_audio: str):
    # 1. Verificar que el archivo existe
    if not os.path.exists(path_audio):
        print(f"No se encontró el archivo: {path_audio}")
        sys.exit(1)

    # 2. Cargar audio
    print(f"\nAnalizando: {path_audio}")
    signal = load_wav(path_audio)

    # 3. Extraer features
    f = extraer_features(signal)
    X = np.array(list(f.values())).reshape(1, -1)

    # 4. Cargar modelo y predecir
    model = joblib.load(MODEL_PATH)
    prediccion    = model.predict(X)[0]
    probabilidad  = model.predict_proba(X).max()

    # 5. Mostrar resultado
    print("─" * 40)
    print(f"Resultado:  {prediccion}")
    print(f"Confianza:  {probabilidad:.1%}")
    print("─" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <ruta/al/audio.wav>")
        print("Ejemplo: python main.py audios/prueba.wav")
        sys.exit(1)

    predecir(sys.argv[1])