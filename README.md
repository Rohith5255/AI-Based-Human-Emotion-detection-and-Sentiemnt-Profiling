# AI-Based Human Emotion Detection and Sentiment Profiling

Multimodal speech emotion recognition on the **MELD** dataset (Friends TV show).
Combines **wav2vec2-base** audio embeddings with **RoBERTa-base** text features via a sigmoid-gated fusion classifier to predict 7 emotions and map them to a 3-class sentiment.

---

## Dataset

**MELD** — Multimodal EmotionLines Dataset
- 13,708 utterances from the Friends TV series
- 7 emotion classes: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`
- Split: 9,989 train / 1,109 dev / 2,610 test
- Severe class imbalance: `neutral` ≈ 47 % of all samples
- Each sample has an audio clip (MP4) and a transcribed utterance

---

## Architecture

```
Audio (MP4/WAV)
    └─ facebook/wav2vec2-base (frozen)
         └─ mean-pool last hidden state → 768-dim
              └─ AudioProjection MLP → 256-dim
                                               ┐
                                               ├─ concat(1024) → sigmoid gate → MLP → 7 classes
                                               ┘
Text (utterance string)
    └─ roberta-base (layers 10-11 + pooler unfrozen)
         └─ CLS token → 768-dim
```

**Sigmoid Gate Fusion**
`fused = concat(audio_256, text_768)` → element-wise `sigmoid(W·fused) * fused` → 3-layer MLP → 7 logits

**Sentiment mapping** (derived from predicted emotion):
- Positive: `joy`
- Negative: `anger`, `disgust`, `fear`, `sadness`
- Neutral: `neutral`, `surprise`

---

## Results

### v2 — MELD, Pre-extracted wav2vec2 Embeddings, Cross-Entropy + Label Smoothing 0.1

| Text Encoder      | Dev WF1 | Test WF1 | Test Acc |
|-------------------|---------|----------|----------|
| BERT-base         | 0.4851  | 0.4861   | 46.7 %   |
| RoBERTa-base      | 0.4793  | 0.5009   | 50.5 %   |
| DistilRoBERTa     | 0.4691  | 0.4915   | 48.8 %   |
| ALBERT-base-v2    | 0.3833  | 0.4306   | 42.2 %   |

### Resumed RoBERTa — 10 extra epochs at LR/3, fresh cosine schedule (best overall)

| Model            | Dev WF1 | Test WF1  | Test Acc  |
|------------------|---------|-----------|-----------|
| RoBERTa resumed  | 0.4825  | **0.5094**| **51.9 %**|

### v3 Ablation — CrossAttention Fusion

| Variant                        | Loss            | RoBERTa Test WF1 |
|-------------------------------|-----------------|------------------|
| v3b: CrossAttn + Focal (γ=2)  | FocalLoss       | ~0.09 (collapsed)|
| v3c: CrossAttn + CE+LS        | CE + LS 0.1     | 0.4961           |
| **v2 resumed (final best)**   | CE + LS 0.1     | **0.5094**        |

> Cross-attention did not improve over sigmoid gate fusion with frozen pre-extracted embeddings.
> Focal loss destabilised training on this dataset.

---

## Project Files

| File | Description |
|------|-------------|
| `cis_project_meld_1.py` | v1: RAVDESS/IEMOCAP baseline, frozen wav2vec2 + text |
| `cis_project_meld_2.py` | v2: Full MELD training, 4 text encoders, sigmoid gate fusion |
| `cis_project_meld_3b.py` | v3b ablation: CrossAttention + Focal Loss (frozen audio) |
| `cis_project_meld_3c.py` | v3c ablation: CrossAttention + CE+LabelSmooth (frozen audio) |
| `cis_project_meld_app.py` | Gradio inference app using best RoBERTa checkpoint |
| `requirements.txt` | Python dependencies |

---

## How to Run the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the MELD checkpoint
Place the trained model at:
```
D:\SEAD\meld_cache\saved_models_meld_v2\best_model_roberta.pth
```

### 3. Launch the Gradio app
```bash
python cis_project_meld_app.py
```
Open `http://127.0.0.1:7860` in your browser.

**Features:**
- **Tab 1 — Upload audio**: upload `.wav`, `.mp3`, `.mp4`, `.m4a`, `.ogg`, or `.flac`. Converted to 16 kHz mono automatically via ffmpeg.
- **Tab 2 — Microphone**: record live and predict.
- **Tab 3 — Text only**: enter a transcript directly (no audio needed).

Outputs: predicted emotion, sentiment (Positive / Negative / Neutral), transcription (via ASR), and a probability bar chart.

---

## How to Run Training (v2)

### 1. Prepare MELD data
Download MELD from [https://github.com/declare-lab/MELD](https://github.com/declare-lab/MELD) and place at:
```
D:\SEAD\meld_cache\MELD.Raw\
```

### 2. Run v2 training
```bash
python cis_project_meld_2.py
```

The script will:
1. Extract wav2vec2 audio embeddings and cache them to `meld_v2_*.pkl`
2. Train 4 text encoders (BERT, RoBERTa, DistilRoBERTa, ALBERT) for 15 epochs each
3. Save best checkpoint per model to `D:\SEAD\meld_cache\saved_models_meld_v2\`
4. Print full classification report on the test set

**Hardware**: tested on NVIDIA GPU with 4 GB VRAM, CUDA 12.x. Training takes ~6-8 hours total.

---

## Dependencies

- Python 3.11
- PyTorch 2.x + CUDA
- HuggingFace Transformers
- librosa, soundfile
- scikit-learn
- Gradio 6.x
- ffmpeg (must be on PATH for audio conversion)
