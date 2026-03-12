from pydub import AudioSegment
from pathlib import Path

def convertir_a_wav(carpeta: str):
    carpeta = Path(carpeta)
    
    for archivo in carpeta.glob("*.m4a"):
        audio = AudioSegment.from_file(archivo)
        destino = archivo.with_suffix(".wav")
        audio.export(destino, format="wav")
        print(f"Convertido: {archivo.name} → {destino.name}")

# Convierte ambas carpetas de una vez
convertir_a_wav("data/FM")
convertir_a_wav("data/WN")


