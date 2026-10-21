"""Аудіофайл -> назва класу та впевненість.

    python src/predict.py file1.wav [file2.mp3 ...] [--threshold 0.7]
"""
import argparse
import json
import warnings

warnings.filterwarnings("ignore")

from common import MODELS, Yamnet


class SoundClassifier:
    def __init__(self, backend="auto", threshold=0.7):
        import tensorflow as tf
        meta = json.load(open(MODELS / "labels.json", encoding="utf-8"))
        self.classes, self.ua = meta["classes"], meta["classes_ua"]
        self.yam = Yamnet(backend)
        self.head = tf.keras.models.load_model(MODELS / "sound_head.keras")
        self.threshold = threshold

    def predict(self, path):
        scores, emb = self.yam.embed_file(path)
        prob = self.head.predict(emb, verbose=0).mean(axis=0)   # усереднення за кадрами
        i = int(prob.argmax())
        return {
            "class": self.classes[i],
            "class_ua": self.ua[self.classes[i]],
            "confidence": float(prob[i]),
            "uncertain": bool(prob[i] < self.threshold),
            "probs": {self.ua[c]: float(p) for c, p in zip(self.classes, prob)},
            "yamnet_top1": self.yam.class_names[int(scores.mean(axis=0).argmax())],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    clf = SoundClassifier(args.backend, args.threshold)
    for f in args.files:
        r = clf.predict(f)
        flag = "  [НИЗЬКА ВПЕВНЕНІСТЬ]" if r["uncertain"] else ""
        print(f"{f}\n  -> {r['class_ua']} ({r['confidence']:.0%}){flag}"
              f"   | YAMNet AudioSet top-1: {r['yamnet_top1']}")


if __name__ == "__main__":
    main()
