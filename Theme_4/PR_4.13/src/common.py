"""Спільні функції: завантаження аудіо та витяг ознак YAMNet."""
from pathlib import Path
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

CLASSES = ["engine", "siren", "rain"]
CLASSES_UA = {"engine": "двигун автомобіля", "siren": "сирена", "rain": "дощ"}

SR = 16000                 # YAMNet очікує моно 16 кГц, значення в [-1, 1]
FRAME_HOP = 0.48           # крок кадру YAMNet, с
SILENCE_RMS = 1e-3         # кадри, тихіші за цей поріг, не використовуються

HUB_URL = "https://tfhub.dev/google/yamnet/1"
TFLITE_PATH = MODELS / "yamnet.tflite"


def load_audio(path, sr=SR):
    """Читає аудіофайл, зводить до моно, передискретизує до 16 кГц."""
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if file_sr != sr:
        g = gcd(int(file_sr), sr)
        wav = resample_poly(wav, sr // g, int(file_sr) // g).astype("float32")
    return np.clip(wav, -1.0, 1.0)


def frame_rms(wav, n_frames):
    """Енергія (RMS) кожного кадру YAMNet (вікно 0.96 с, крок 0.48 с)."""
    hop, win = int(FRAME_HOP * SR), int(0.96 * SR)
    out = []
    for i in range(n_frames):
        seg = wav[i * hop: i * hop + win]
        out.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    return np.array(out)


class Yamnet:
    """Обгортка над готовою YAMNet.

    backend="hub"    — оригінальна модель з TensorFlow Hub (потрібен Інтернет);
    backend="tflite" — та сама модель TF Hub у форматі TFLite (models/yamnet.tflite),
                       працює офлайн;
    backend="auto"   — спершу локальний TFLite, якщо файлу немає — TF Hub.
    """

    def __init__(self, backend="auto"):
        import tensorflow as tf
        self.tf = tf
        if backend == "auto":
            backend = "tflite" if TFLITE_PATH.exists() else "hub"
        self.backend = backend
        if backend == "hub":
            import tensorflow_hub as hub
            self.model = hub.load(HUB_URL)
        else:
            self.interp = tf.lite.Interpreter(str(TFLITE_PATH))
            self._in = self.interp.get_input_details()[0]["index"]
            outs = self.interp.get_output_details()
            # виходи: scores [N,521], embeddings [N,1024], spectrogram
            self._out = {o["shape"][-1]: o["index"] for o in outs}
        self.class_names = [l.strip() for l in open(MODELS / "yamnet_class_map.txt", encoding="utf-8")]

    def __call__(self, wav):
        """Повертає (scores [N,521], embeddings [N,1024]) для кожного кадру."""
        wav = np.asarray(wav, dtype="float32")
        if self.backend == "hub":
            scores, emb, _ = self.model(wav)
            return scores.numpy(), emb.numpy()
        self.interp.resize_tensor_input(self._in, [len(wav)], strict=False)
        self.interp.allocate_tensors()
        self.interp.set_tensor(self._in, wav)
        self.interp.invoke()
        return (self.interp.get_tensor(self._out[521]).copy(),
                self.interp.get_tensor(self._out[1024]).copy())

    def embed_file(self, path, drop_silence=True):
        """Ознаки кадрів файлу; тихі кадри (паузи у записі) відкидаються."""
        wav = load_audio(path)
        scores, emb = self(wav)
        if drop_silence:
            keep = frame_rms(wav, len(emb)) > SILENCE_RMS
            if keep.any():
                scores, emb = scores[keep], emb[keep]
        return scores, emb
