import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.dataset    import load_dataset
from src.processor   import extraer_features
import pandas as pd

def save_dataset(data_dir='data'):
    signals, labels, names = load_dataset(data_dir)
    
    filas = []
    for signal, label, nombre in zip(signals, labels, names):
        f = extraer_features(signal)
        f["archivo"] = nombre
        f["label"]   = label
        filas.append(f)
    
    df = pd.DataFrame(filas)
    df.to_csv(f"{data_dir}/dataset.csv", index=False)
    print(f"Dataset guardado: {len(df)} filas")
    print(df.head())
    return df

if __name__ == "__main__":
    save_dataset('data')