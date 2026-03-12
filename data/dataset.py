import os
import numpy as np
import soundfile as sf

N_SAMPLES = 22_050  

def load_wav(filepath):
    signal, sr = sf.read(filepath)
    
    # Mono si es estéreo
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    
    # Convertir a float32 y restar media
    signal = signal.astype(np.float32)
    signal -= signal.mean()
    
    # Ajustar longitud
    if len(signal) >= N_SAMPLES:
        return signal[:N_SAMPLES]
    return np.pad(signal, (0, N_SAMPLES - len(signal)))

def load_dataset(data_dir='data'):
    signals, labels, names = [], [], []

    for folder, label in [('fm', 'FM Radio'), ('wn', 'White Noise')]:
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            print(f"⚠ Carpeta no encontrada: {path}")
            continue
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith('.wav'):
                try:
                    signals.append(load_wav(os.path.join(path, fname)))
                    labels.append(label)
                    names.append(fname)
                except Exception as e:
                    print(f"✗ Error leyendo {fname}: {e}")

    return signals, labels, names
