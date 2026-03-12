# fm-noise-classifier 🎙️

Signal classifier that distinguishes FM radio signals from white noise using autocorrelation, FFT and norm analysis in Python.

---

## 📋 Description

This project implements a machine learning pipeline to classify audio signals as either **FM Radio** or **White Noise**. It uses digital signal processing techniques to extract meaningful features from audio files, trains a Random Forest classifier, and exposes a simple CLI to predict any new audio file.

---

## 🧠 How it works

```
Audio (.wav / .m4a)
       ↓
1. Load & center signal     → subtract mean
       ↓
2. Autocorrelation          → does the signal repeat itself?
       ↓
3. FFT features             → spectral entropy + spectral flatness
       ↓
4. Norm features            → crest factor + kurtosis
       ↓
5. Random Forest            → FM Radio or White Noise?
```

### Features extracted per signal

| Feature | Description | FM Radio | White Noise |
|---|---|---|---|
| `autocorr_peak` | Max autocorrelation peak | High | ≈ 0 |
| `entropy` | Spectral entropy | Low | ≈ 1.0 |
| `flatness` | Spectral flatness | Low | ≈ 1.0 |
| `crest_factor` | Peak / RMS ratio | ≈ 1.41 | High |
| `kurtosis` | Signal distribution shape | < 2.5 | ≈ 3.0 |

---

## 📁 Project Structure

```
fm-noise-classifier/
├── data/
│   ├── fm/                  ← FM radio recordings (.wav)
│   ├── WN/                  ← White noise recordings (.wav)
│   ├── AudioPrueba/         ← Test audio files
│   │   ├── FM/
│   │   └── WN/
│   ├── dataset.csv          ← Generated feature dataset
│   ├── dataset.py           ← Audio loader
│   └── __init__.py
├── model/
│   ├── train.py             ← Train and save the model
│   ├── modelo.pkl           ← Trained model (generated)
│   └── __init__.py
├── src/
│   ├── processor.py         ← Feature extraction
│   └── __init__.py
├── utils/
│   ├── save_dataset.py      ← Generate dataset.csv (run once)
│   ├── convertir_audio.py   ← Convert .m4a to .wav (run once)
│   └── renombrar_archivos.py ← Rename files (run once)
└── main.py                  ← CLI predictor
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/fm-noise-classifier.git
cd fm-noise-classifier
pip install numpy scipy scikit-learn pandas soundfile pydub joblib
```

---

## 🚀 Usage

### 1. Prepare your audio files
Place your recordings in:
- `data/fm/` → FM radio recordings
- `data/WN/` → White noise recordings

### 2. Generate the dataset (run once)
```bash
python utils/save_dataset.py
```

### 3. Train the model (run once)
```bash
python model/train.py
```

### 4. Predict a new audio file
```bash
python main.py path/to/audio.wav
```

Output:
```
Analizando: path/to/audio.wav
────────────────────────────────────────
Resultado:  FM Radio
Confianza:  90.0%
────────────────────────────────────────
```

---

## 📊 Model Performance

Trained on 100 audio samples (50 FM + 50 White Noise), 80/20 train-test split.

```
              precision    recall  f1-score
   FM Radio       0.67      1.00      0.80
White Noise       1.00      0.50      0.67

   accuracy                           0.75
```

> ⚠️ Accuracy can be improved by adding more White Noise samples with greater variety.

---

## 🔧 Tech Stack

- **Python 3.12**
- **numpy / scipy** — signal processing
- **scikit-learn** — Random Forest classifier
- **soundfile / pydub** — audio loading
- **pandas** — dataset management
- **joblib** — model persistence

---

## 📌 Roadmap

- [x] Audio loading and preprocessing
- [x] Feature extraction (autocorrelation, FFT, norm)
- [x] Random Forest classifier
- [x] CLI prediction
- [ ] Add more White Noise samples to improve accuracy
- [ ] Reach 90%+ accuracy
- [ ] Support batch prediction from folder

---

## 👨‍💻 Author

Built as an academic project for **Procesos Estocásticos**.
