import numpy as np
from scipy.signal import correlate
from scipy.fft import fft

 
def autocorrelation(x):
    """¿La señal se repite? FM sí, ruido no."""
    r = correlate(x, x, mode='full')
    r = r[len(r) // 2:]
    r /= r[0] + 1e-12
    peak = float(np.max(np.abs(r[1:len(x) // 2])))
    return peak
 
def fft_features(x):
    """¿Tiene frecuencias ordenadas? FM sí, ruido no."""
    X   = np.abs(fft(x))[:len(x) // 2]
    psd = X ** 2
    psd_norm = psd / (psd.sum() + 1e-12)
 
    entropy  = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(len(psd_norm)))
    flatness = float(np.exp(np.mean(np.log(psd + 1e-12))) / (psd.mean() + 1e-12))
    return entropy, flatness
 
def norm_features(x):
    """¿Cómo son sus picos? Sinusoide FM vs ruido impulsivo."""
    rms          = np.sqrt(np.mean(x ** 2))
    crest_factor = float(np.max(np.abs(x)) / (rms + 1e-12))
    kurtosis     = float(np.mean(((x - x.mean()) / (x.std() + 1e-12)) ** 4))
    return crest_factor, kurtosis

def extraer_features(x: np.ndarray) -> dict:
    peak                    = autocorrelation(x)
    entropy, flatness       = fft_features(x)
    crest, kurtosis         = norm_features(x)
    
    return {
        "autocorr_peak" : peak,
        "entropy"       : entropy,
        "flatness"      : flatness,
        "crest_factor"  : crest,
        "kurtosis"      : kurtosis
    }