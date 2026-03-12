import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import classification_report, confusion_matrix

def train(dataset_path='data/dataset.csv', model_path='model/modelo.pkl'):
    
    # 1. Cargar dataset
    print("Cargando dataset…")
    df = pd.read_csv(dataset_path)
    
    # 2. Separar features y etiquetas
    FEATURES = ["autocorr_peak", "entropy", "flatness", "crest_factor", "kurtosis"]
    X = df[FEATURES].values
    y = df["label"].values
    
    # 3. Dividir en entrenamiento y prueba (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Prueba:        {len(X_test)} muestras\n")
    
    # 4. Entrenar modelo
    print("Entrenando modelo…")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluar
    y_pred = model.predict(X_test)
    print("─" * 40)
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("─" * 40)
    
    # 6. Guardar modelo
    joblib.dump(model, model_path)
    print(f"\nModelo guardado en: {model_path}")

if __name__ == "__main__":
    train()
