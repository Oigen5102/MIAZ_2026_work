"""Відбір навчальних і тестових записів з відкритого набору ESC-50.

ESC-50 (K. J. Piczak, 2015): 2000 записів по 5 с, 50 класів, 5 фолдів, ліцензія CC BY-NC 3.0.
https://github.com/karolpiczak/ESC-50

Використання:
    git clone --depth 1 https://github.com/karolpiczak/ESC-50
    python src/prepare_data.py --esc50 ESC-50
"""
import argparse
import csv
import shutil
from pathlib import Path

import soundfile as sf

from common import CLASSES, DATA, SR, load_audio

TRAIN_FOLDS = {1, 2}       # навчання: 15 записів на клас
TEST_FOLDS = {3, 4, 5}     # перевірка: нові записи (інші джерела), 24 на клас
N_TRAIN = 15
# Звуки поза трьома класами — щоб показати типову помилку «закритого світу»
OOD = ["thunderstorm", "helicopter", "train", "water_drops"]
N_OOD = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", required=True, help="шлях до клону ESC-50")
    args = ap.parse_args()
    esc = Path(args.esc50)
    rows = list(csv.DictReader(open(esc / "meta" / "esc50.csv")))

    for sub in ("train", "test", "ood"):
        shutil.rmtree(DATA / sub, ignore_errors=True)

    manifest = []

    def put(row, split, label):
        dst = DATA / split / label / row["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, load_audio(esc / "audio" / row["filename"]), SR, subtype="PCM_16")
        manifest.append({"split": split, "label": label, "file": str(dst.relative_to(DATA)),
                         "esc50_file": row["filename"], "fold": row["fold"]})

    for c in CLASSES:
        cls = sorted((r for r in rows if r["category"] == c), key=lambda r: r["filename"])
        for r in [r for r in cls if int(r["fold"]) in TRAIN_FOLDS][:N_TRAIN]:
            put(r, "train", c)
        for r in [r for r in cls if int(r["fold"]) in TEST_FOLDS]:
            put(r, "test", c)
    for c in OOD:
        for r in [r for r in rows if r["category"] == c and int(r["fold"]) in TEST_FOLDS][:N_OOD]:
            put(r, "ood", c)

    with open(DATA / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    for s in ("train", "test", "ood"):
        print(s, sum(m["split"] == s for m in manifest), "файлів")


if __name__ == "__main__":
    main()
