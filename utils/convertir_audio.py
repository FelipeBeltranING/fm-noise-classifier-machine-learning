from pathlib import Path

def renombrar_archivos(carpeta: str, prefijo: str):
    carpeta = Path(carpeta)
    archivos = sorted(carpeta.glob("*.wav"))
    
    for i, archivo in enumerate(archivos, start=1):
        nuevo_nombre = carpeta / f"{prefijo}_{i:02d}.wav"
        archivo.rename(nuevo_nombre)
        print(f"{archivo.name} → {nuevo_nombre.name}")

renombrar_archivos("data/fm", "fm")
renombrar_archivos("data/WN", "wn")
