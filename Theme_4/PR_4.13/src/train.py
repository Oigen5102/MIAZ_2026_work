"""Transfer learning: готові ознаки YAMNet (1024) + новий класифікатор на 3 класи.

Ваги YAMNet заморожені — навчається лише невелика «голова» (Dense-шари).
Для порівняння оцінюється YAMNet «як є» (zero-shot) через її власні класи AudioSet.

    python src/train.py [--backend auto|tflite|hub]
"""
import argparse
import warnings
import csv
import json

import numpy as np

warnings.filterwarnings("ignore")

from common import CLASSES, CLASSES_UA, DATA, MODELS, RESULTS, Yamnet

SEED = 42
EPOCHS = 40

# Класи AudioSet (з 521 класу YAMNet), що відповідають нашим трьом
ZERO_SHOT_MAP = {
    "engine": ["Engine", "Light engine (high frequency)", "Medium engine (mid frequency)",
               "Heavy engine (low frequency)", "Engine knocking", "Engine starting", "Idling",
               "Accelerating, revving, vroom", "Car", "Motor vehicle (road)", "Vehicle"],
    "siren": ["Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)",
              "Fire engine, fire truck (siren)", "Emergency vehicle"],
    "rain": ["Rain", "Raindrop", "Rain on surface"],
}


def files(split):
    return [(p, c) for c in CLASSES for p in sorted((DATA / split / c).glob("*.wav"))]


def build_head(n_classes):
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,), name="yamnet_embedding"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="sound_head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    yam = Yamnet(args.backend)
    print("YAMNet backend:", yam.backend)
    RESULTS.mkdir(exist_ok=True)

    # 1. Ознаки YAMNet для навчальних записів (кожен кадр 0.96 с -> вектор 1024)
    X, y = [], []
    for p, c in files("train"):
        _, emb = yam.embed_file(p)
        X.append(emb)
        y += [CLASSES.index(c)] * len(emb)
    X, y = np.concatenate(X), np.array(y)
    print(f"Навчальних записів: {len(files('train'))}, кадрів-ознак: {X.shape}")

    # 2. Навчання нового класифікатора (ваги YAMNet не змінюються)
    head = build_head(len(CLASSES))
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = head.fit(X, y, epochs=EPOCHS, batch_size=32, shuffle=True, verbose=0)
    print(f"train acc (кадри): {hist.history['accuracy'][-1]:.3f}")
    head.save(MODELS / "sound_head.keras")
    json.dump({"classes": CLASSES, "classes_ua": CLASSES_UA, "backend": yam.backend},
              open(MODELS / "labels.json", "w"), ensure_ascii=False, indent=1)

    # 3. Оцінка на нових записах: середнє ймовірностей за кадрами запису
    idx = {c: [yam.class_names.index(n) for n in names] for c, names in ZERO_SHOT_MAP.items()}
    rows, y_true, y_head, y_zero = [], [], [], []
    for p, c in files("test"):
        scores, emb = yam.embed_file(p)
        prob = head.predict(emb, verbose=0).mean(axis=0)
        zs = np.array([scores[:, idx[k]].max(axis=1).mean() for k in CLASSES])
        top = yam.class_names[int(scores.mean(axis=0).argmax())]
        y_true.append(c); y_head.append(CLASSES[prob.argmax()]); y_zero.append(CLASSES[zs.argmax()])
        rows.append({"file": p.name, "true": c, "pred": y_head[-1], "conf": round(float(prob.max()), 3),
                     **{f"p_{k}": round(float(v), 3) for k, v in zip(CLASSES, prob)},
                     "zero_shot": y_zero[-1], "yamnet_top1": top})

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    acc_head, acc_zero = accuracy_score(y_true, y_head), accuracy_score(y_true, y_zero)
    cm = confusion_matrix(y_true, y_head, labels=CLASSES)
    print(f"\nТест ({len(y_true)} нових записів): точність класифікатора = {acc_head:.3f}, "
          f"YAMNet zero-shot = {acc_zero:.3f}")
    print(classification_report(y_true, y_head, labels=CLASSES, digits=3))
    print("Матриця помилок (рядки — істина):\n", cm)

    with open(RESULTS / "test_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump({"backend": yam.backend, "n_train_clips": len(files("train")), "n_train_frames": int(len(X)),
               "n_test_clips": len(y_true), "accuracy_head": acc_head, "accuracy_zero_shot": acc_zero,
               "confusion_matrix": cm.tolist(), "classes": CLASSES,
               "report": classification_report(y_true, y_head, labels=CLASSES, output_dict=True),
               "errors": [r for r in rows if r["true"] != r["pred"]]},
              open(RESULTS / "metrics.json", "w"), ensure_ascii=False, indent=1)
    plot(cm, hist.history, acc_head, acc_zero)


def plot(cm, hist, acc_head, acc_zero):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [CLASSES_UA[c] for c in CLASSES]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks(range(3), labels, rotation=15); ax.set_yticks(range(3), labels)
    ax.set_xlabel("Прогноз"); ax.set_ylabel("Істина")
    ax.set_title(f"Матриця помилок, точність {acc_head:.1%}")
    fig.tight_layout(); fig.savefig(RESULTS / "confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(hist["loss"], label="loss"); ax.plot(hist["accuracy"], label="accuracy")
    ax.set_xlabel("епоха"); ax.legend(); ax.set_title("Навчання класифікатора (кадри)")
    fig.tight_layout(); fig.savefig(RESULTS / "training_curve.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
