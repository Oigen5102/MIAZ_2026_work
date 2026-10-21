"""Веб-застосунок: завантажте або запишіть звук -> клас (двигун / сирена / дощ).

    python app.py          # відкрити http://127.0.0.1:7860
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from predict import SoundClassifier

clf = SoundClassifier()
examples = sorted(str(p) for p in (Path(__file__).parent / "demo").glob("*.wav"))


def classify(path):
    if not path:
        return {}, ""
    r = clf.predict(path)
    note = (f"**{r['class_ua']}** — впевненість {r['confidence']:.0%}"
            + ("  \n⚠️ Низька впевненість: можливо, звук не належить до жодного з трьох класів."
               if r["uncertain"] else "")
            + f"  \nYAMNet (AudioSet, без донавчання): *{r['yamnet_top1']}*")
    return r["probs"], note


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(type="filepath", label="Аудіофайл (wav/mp3) або запис з мікрофона"),
    outputs=[gr.Label(num_top_classes=3, label="Клас"), gr.Markdown()],
    examples=examples or None,
    title="Розпізнавання звуків: двигун / сирена / дощ",
    description="YAMNet (TensorFlow Hub) як готовий екстрактор ознак + навчений класифікатор на 3 класи.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
