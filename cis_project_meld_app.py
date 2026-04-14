# -*- coding: utf-8 -*-
"""
CIS Project — Multimodal SER Inference App (Gradio)
====================================================
End product: users upload audio, record via microphone, or type text to get
  • Predicted emotion  (anger / disgust / fear / joy / neutral / sadness / surprise)
  • Sentiment label    (Positive / Negative / Neutral)
  • Per-class probability bar chart (emotion-specific colours)

Inference pipeline
------------------
  Audio  → wav2vec2-base (frozen, mean-pool)  → 768-dim embed
  Audio  → wav2vec2-base-960h (ASR)           → transcript text
  Text   → roberta-base (top-2-layers fine-tuned) → CLS 768-dim
         → AudioProjection MLP (768 → 256)
  Concat (256 + 768 = 1024) → sigmoid gate → Classifier → 7 logits
  → softmax → emotion + sentiment

Text-only mode: audio embedding is a zero-vector (768-dim zeros).
  The text signal alone drives the prediction in this mode.

Checkpoint: D:\\SEAD\\meld_cache\\saved_models_meld_v2\\best_model_roberta.pth
  (v2 best + 10 resume epochs at LR/3 — epoch 13 + 10)
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
import librosa
import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModel,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from sklearn.preprocessing import LabelEncoder

# ── ffmpeg for mp4/mp3 decoding ─────────────────────────────────────────────
_ffmpeg_bin = (
    r"C:\Users\rohit\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
)
if _ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = r"D:\SEAD\meld_cache\saved_models_meld_v2\best_model_roberta.pth"

# ── Constants (must match v2 training script exactly) ────────────────────────
TARGET_SR     = 16000
WAV2VEC_MODEL = "facebook/wav2vec2-base"
WAV2VEC_DIM   = 768
ASR_MODEL     = "facebook/wav2vec2-base-960h"

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)

TEXT_MODELS = {
    "roberta": {
        "name":      "roberta-base",
        "embed_dim": 768,
        "description": "RoBERTa base",
    },
}

SENTIMENT_MAP = {
    "joy":      "Positive",
    "surprise": "Positive",
    "neutral":  "Neutral",
    "anger":    "Negative",
    "disgust":  "Negative",
    "fear":     "Negative",
    "sadness":  "Negative",
}

SENTIMENT_EMOJI = {
    "Positive": "😊 Positive",
    "Negative": "😞 Negative",
    "Neutral":  "😐 Neutral",
}

EMOTION_EMOJI = {
    "anger":    "😠 Anger",
    "disgust":  "🤢 Disgust",
    "fear":     "😨 Fear",
    "joy":      "😄 Joy",
    "neutral":  "😐 Neutral",
    "sadness":  "😢 Sadness",
    "surprise": "😲 Surprise",
}

# Emotion-specific colours for the probability bar chart
EMOTION_COLORS = {
    "anger":    "#e74c3c",   # red
    "disgust":  "#8B4513",   # brown
    "fear":     "#9b59b6",   # purple
    "joy":      "#2ecc71",   # green
    "neutral":  "#95a5a6",   # gray
    "sadness":  "#3498db",   # blue
    "surprise": "#e67e22",   # orange
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── HTML bar chart with emotion-specific colours ──────────────────────────────

def make_emotion_chart_html(probs_array):
    """Styled probability bar chart with emotion-specific colours."""
    items = sorted(zip(EMOTIONS, probs_array), key=lambda x: x[1], reverse=True)
    rows = ""
    for emotion, prob in items:
        pct   = prob * 100
        emoji = EMOTION_EMOJI[emotion].split()[0]
        name  = EMOTION_EMOJI[emotion].split()[1] if len(EMOTION_EMOJI[emotion].split()) > 1 else emotion.title()
        color = EMOTION_COLORS[emotion]
        wide  = " wide" if pct >= 18 else ""
        rows += (
            f'<div class="prob-row">'
            f'  <div class="prob-label-row">'
            f'    <span class="prob-emo-label">{emoji} {name}</span>'
            f'    <span class="prob-pct-label">{pct:.1f}%</span>'
            f'  </div>'
            f'  <div class="prob-track">'
            f'    <div class="prob-fill{wide}" style="background:{color};width:{pct:.1f}%">'
            f'      <span class="prob-fill-label">{pct:.0f}%</span>'
            f'    </div>'
            f'  </div>'
            f'</div>'
        )
    return (
        '<div style="padding:14px 12px 10px;font-family:\'Inter\',sans-serif">'
        '<div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;'
        'letter-spacing:0.6px;margin-bottom:14px">Emotion Probabilities</div>'
        + rows +
        '</div>'
    )


# ── Audio helper ─────────────────────────────────────────────────────────────

def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    min_len = TARGET_SR // 2          # pad to at least 0.5 s
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    return audio


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s']", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "unknown"


# ── Model architecture (identical to cis_project_meld_2.py) ─────────────────

class AudioProjection(nn.Module):
    def __init__(self, input_dim=WAV2VEC_DIM, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self, model_key):
        super().__init__()
        cfg            = TEXT_MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
        self.model     = AutoModel.from_pretrained(cfg["name"], use_safetensors=True)
        self.embed_dim = cfg["embed_dim"]
        self.model_key = model_key

        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze top 2 layers + pooler (matches training config)
        for name, param in self.model.named_parameters():
            if any(k in name for k in [
                "layer.10", "layer.11",
                "encoder.layer.10", "encoder.layer.11",
                "pooler",
            ]):
                param.requires_grad = True

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out    = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]   # CLS token


class MultimodalFusion(nn.Module):
    def __init__(self, audio_dim=256, text_dim=768, hidden_dim=512, num_classes=7):
        super().__init__()
        fusion_dim = audio_dim + text_dim        # 1024
        self.gate  = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, audio_vec, text_vec):
        fused  = torch.cat((audio_vec, text_vec), dim=1)
        gate_w = self.gate(fused)
        return self.classifier(fused * gate_w)


class EmotionModel(nn.Module):
    def __init__(self, model_key="roberta"):
        super().__init__()
        self.audio_proj = AudioProjection()
        self.text_enc   = TextEncoder(model_key)
        self.fusion     = MultimodalFusion(
            text_dim=self.text_enc.embed_dim,
            num_classes=len(EMOTIONS),
        )

    def forward(self, audio_emb, texts):
        a = self.audio_proj(audio_emb)    # (B, 256)
        t = self.text_enc(texts)          # (B, 768)
        return self.fusion(a, t)


# ── wav2vec2 audio feature extractor (frozen) ────────────────────────────────

class Wav2VecExtractor:
    def __init__(self):
        print(f"  Loading {WAV2VEC_MODEL} for audio features ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL)
        self.model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("  wav2vec2-base loaded.\n")

    @torch.no_grad()
    def embed(self, audio_np):
        """Return mean-pooled last hidden state: shape (768,)."""
        inputs = self.feature_extractor(
            audio_np, sampling_rate=TARGET_SR,
            return_tensors="pt", padding=True,
        )
        out = self.model(inputs.input_values.to(device))
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


# ── ASR: speech → text ───────────────────────────────────────────────────────

class ASRModel:
    def __init__(self):
        print(f"  Loading {ASR_MODEL} for ASR ...")
        self.processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL)
        self.model     = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL).to(device)
        self.model.eval()
        print("  ASR model loaded.\n")

    @torch.no_grad()
    def transcribe(self, audio_np):
        inputs = self.processor(
            audio_np, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        logits    = self.model(inputs.input_values.to(device)).logits
        pred_ids  = torch.argmax(logits, dim=-1)
        transcript = self.processor.batch_decode(pred_ids)[0]
        return transcript.lower().strip()


# ── Load models at startup ────────────────────────────────────────────────────

print("=" * 55)
print("  Loading models — this may take ~30 seconds ...")
print("=" * 55)

wav2vec_extractor = Wav2VecExtractor()
asr_model         = ASRModel()

print("  Loading trained EmotionModel checkpoint ...")
emotion_model = EmotionModel("roberta").to(device)
emotion_model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
)
emotion_model.eval()
print(f"  Checkpoint loaded: {CHECKPOINT_PATH}")
print("All models ready.\n")


# ── ffmpeg conversion: any format → wav ──────────────────────────────────────

def convert_to_wav(input_path):
    """
    Convert any audio/video file to a temp 16kHz mono WAV using ffmpeg.
    Returns (path, is_temp). Caller is responsible for deleting the temp file.
    If the file is already a .wav, returns the original path unchanged.
    """
    import tempfile
    import subprocess
    if input_path.lower().endswith(".wav"):
        return input_path, False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(TARGET_SR), "-ac", "1",
        "-f", "wav", tmp.name
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed:\n{result.stderr.decode(errors='replace')}"
        )
    return tmp.name, True


# ── Core inference ────────────────────────────────────────────────────────────

def run_inference(audio_np, text):
    """
    audio_np : np.ndarray at TARGET_SR, or None (text-only mode)
    text      : str — can be empty string
    Returns   : (emotion_str, sentiment_str, transcript_str, chart_html_str, probs_array)
    """
    if audio_np is not None:
        audio_emb = wav2vec_extractor.embed(audio_np)
    else:
        audio_emb = np.zeros(WAV2VEC_DIM, dtype=np.float32)

    transcript = ""
    if audio_np is not None and not text.strip():
        transcript     = asr_model.transcribe(audio_np)
        text_for_model = clean_text(transcript)
    else:
        transcript     = text.strip()
        text_for_model = clean_text(text.strip()) if text.strip() else "unknown"

    audio_tensor = torch.tensor(audio_emb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = emotion_model(audio_tensor, [text_for_model])
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    emotion    = EMOTIONS[int(probs.argmax())]
    sentiment  = SENTIMENT_MAP[emotion]
    chart_html = make_emotion_chart_html(probs)

    return emotion, sentiment, transcript, chart_html, probs


# ── Result HTML helpers ───────────────────────────────────────────────────────

def make_emotion_card_html(emotion, probs_array):
    """Large centred card: big emoji + name + confidence %."""
    if emotion in ("—", "Error", None):
        return (
            '<div class="emotion-card emotion-card-empty">'
            '  <div class="emotion-emoji">🎯</div>'
            '  <div class="emotion-card-name" style="color:#94a3b8">Awaiting input</div>'
            '  <div class="emotion-card-conf">—</div>'
            '</div>'
        )
    parts = EMOTION_EMOJI[emotion].split(maxsplit=1)
    emoji, name = parts[0], parts[1] if len(parts) > 1 else emotion.title()
    conf  = float(probs_array[EMOTIONS.index(emotion)]) * 100
    color = EMOTION_COLORS[emotion]
    return (
        f'<div class="emotion-card">'
        f'  <div class="emotion-emoji">{emoji}</div>'
        f'  <div class="emotion-card-name" style="color:{color}">{name}</div>'
        f'  <div class="emotion-card-conf">{conf:.1f}% confidence</div>'
        f'</div>'
    )


def make_sentiment_badge_html(sentiment):
    """Coloured pill badge for sentiment."""
    cfg = {
        "Positive": ("#dcfce7", "#15803d", "😊 Positive"),
        "Negative": ("#fee2e2", "#dc2626", "😞 Negative"),
        "Neutral":  ("#f1f5f9", "#475569", "😐 Neutral"),
    }
    if sentiment not in cfg:
        return (
            '<div class="sentiment-wrapper">'
            '  <span class="sentiment-badge" style="background:#f1f5f9;color:#94a3b8">—</span>'
            '</div>'
        )
    bg, fg, label = cfg[sentiment]
    return (
        f'<div class="sentiment-wrapper">'
        f'  <span class="sentiment-badge" style="background:{bg};color:{fg};'
        f'        border:1.5px solid {fg}33">{label}</span>'
        f'</div>'
    )


def make_transcript_html(text):
    """Terminal-style transcript block."""
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<div class="transcript-block">'
        '  <div class="transcript-label">▶ ASR TRANSCRIPT</div>'
        f'  <div class="transcript-text">{safe}</div>'
        '</div>'
    )


_EMPTY_CHART     = make_emotion_chart_html(np.zeros(len(EMOTIONS), dtype=np.float32))
_EMPTY_CARD      = make_emotion_card_html("—", np.zeros(len(EMOTIONS)))
_EMPTY_SENTIMENT = make_sentiment_badge_html("—")
_EMPTY_TRANSCRIPT = make_transcript_html("(no transcription yet)")


# ── Gradio callback functions ─────────────────────────────────────────────────

def predict_upload(file_obj):
    if file_obj is None:
        return _EMPTY_CARD, _EMPTY_SENTIMENT, _EMPTY_TRANSCRIPT, _EMPTY_CHART

    audio_path = file_obj.name
    if os.path.basename(audio_path).startswith("._") or "._" in os.path.basename(audio_path):
        return _EMPTY_CARD, _EMPTY_SENTIMENT, make_transcript_html("Please upload a valid audio file."), _EMPTY_CHART

    try:
        with open(audio_path, "rb") as fh:
            header = fh.read(4)
    except OSError as e:
        return _EMPTY_CARD, _EMPTY_SENTIMENT, make_transcript_html(f"Cannot read file: {e}"), _EMPTY_CHART

    if len(header) < 4:
        return _EMPTY_CARD, _EMPTY_SENTIMENT, make_transcript_html("Please upload a valid audio file."), _EMPTY_CHART

    return predict_audio(audio_path)


def predict_audio(audio_path):
    if audio_path is None:
        return _EMPTY_CARD, _EMPTY_SENTIMENT, _EMPTY_TRANSCRIPT, _EMPTY_CHART

    wav_path, is_temp = None, False
    try:
        wav_path, is_temp = convert_to_wav(audio_path)
        audio_np = load_audio(wav_path)
    except RuntimeError as e:
        return _EMPTY_CARD, _EMPTY_SENTIMENT, make_transcript_html(f"ffmpeg error: {e}"), _EMPTY_CHART
    except Exception:
        return _EMPTY_CARD, _EMPTY_SENTIMENT, make_transcript_html("Unsupported or corrupt file."), _EMPTY_CHART
    finally:
        if is_temp and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

    emotion, sentiment, transcript, chart_html, probs = run_inference(audio_np, "")
    transcript_text = transcript if transcript else "(ASR returned empty)"

    return (
        make_emotion_card_html(emotion, probs),
        make_sentiment_badge_html(sentiment),
        make_transcript_html(transcript_text),
        chart_html,
    )


def predict_text(text_input):
    if not text_input or not text_input.strip():
        return _EMPTY_CARD, _EMPTY_SENTIMENT, _EMPTY_CHART

    emotion, sentiment, _, chart_html, probs = run_inference(None, text_input)

    return (
        make_emotion_card_html(emotion, probs),
        make_sentiment_badge_html(sentiment),
        chart_html,
    )


# ── Custom CSS ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Base ─────────────────────────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: #eef0f8 !important;
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 16px !important;
}

/* ── App Header ─────────────────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1a1f36 0%, #2d3561 100%);
    padding: 36px 40px 30px;
    border-radius: 16px;
    margin-bottom: 16px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(26,31,54,0.3);
}
.app-title {
    font-size: 30px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}
.app-subtitle {
    font-size: 12px;
    color: #a5b4fc;
    margin-bottom: 22px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    font-weight: 500;
}
.metric-pills {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
}
.metric-pill {
    background: rgba(99,102,241,0.22);
    border: 1px solid rgba(99,102,241,0.45);
    color: #e0e7ff;
    padding: 5px 15px;
    border-radius: 24px;
    font-size: 12px;
    font-weight: 600;
}
.metric-pill-label {
    color: #a5b4fc;
    font-weight: 400;
    margin-right: 4px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Stats Table ─────────────────────────────────────────────────────────────── */
.stats-wrap {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}
.stats-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.stats-table th {
    background: #f8fafc;
    color: #64748b;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 10px 16px;
    border-bottom: 1px solid #e2e8f0;
    text-align: left;
}
.stats-table td {
    padding: 12px 16px;
    color: #1e293b;
    font-weight: 500;
    border-bottom: none;
}
.stats-badge {
    background: #ede9fe;
    color: #5b21b6;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 700;
}

/* ── Tabs ────────────────────────────────────────────────────────────────────── */
.tab-nav {
    background: white !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 4px !important;
    display: flex !important;
    border: 1px solid #e2e8f0 !important;
    border-bottom: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    margin-bottom: 12px !important;
}
.tab-nav > button {
    border-radius: 8px !important;
    padding: 9px 20px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border: none !important;
    background: transparent !important;
    color: #64748b !important;
    transition: all 0.18s ease !important;
    flex: 1 !important;
}
.tab-nav > button:hover {
    background: #f1f5f9 !important;
    color: #1e293b !important;
}
.tab-nav > button.selected {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.35) !important;
}

/* ── Cards / Blocks ─────────────────────────────────────────────────────────── */
.block, .gr-box, .gr-form, .form {
    background: white !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* ── Primary Button ─────────────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 13px 24px !important;
    color: white !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    margin-top: 8px !important;
}
button.primary:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}
button.primary:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
}

/* ── Component Labels ───────────────────────────────────────────────────────── */
.block > label > span, .label-wrap span {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

/* ── Emotion Result Card ────────────────────────────────────────────────────── */
.emotion-card {
    text-align: center;
    padding: 28px 20px 22px;
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    min-height: 140px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.emotion-card-empty { opacity: 0.38; }
.emotion-emoji {
    font-size: 54px;
    line-height: 1.1;
    margin-bottom: 8px;
}
.emotion-card-name {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
    font-family: 'Inter', sans-serif;
}
.emotion-card-conf {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 600;
    letter-spacing: 0.2px;
}

/* ── Sentiment Badge ────────────────────────────────────────────────────────── */
.sentiment-wrapper {
    text-align: center;
    padding: 18px 12px;
}
.sentiment-badge {
    display: inline-flex;
    align-items: center;
    padding: 10px 24px;
    border-radius: 24px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.2px;
}

/* ── Transcript Terminal Block ──────────────────────────────────────────────── */
.transcript-block {
    background: #0f172a;
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid #1e293b;
    margin: 4px 0;
}
.transcript-label {
    font-size: 9px;
    font-weight: 700;
    color: #475569;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    font-family: 'Consolas', monospace;
}
.transcript-text {
    font-family: 'Consolas', 'Fira Code', 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #7dd3fc;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Probability Bars ───────────────────────────────────────────────────────── */
.prob-row { margin-bottom: 11px; }
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.prob-emo-label { font-size: 13px; font-weight: 500; font-family: 'Inter', sans-serif; }
.prob-pct-label { font-size: 12px; font-weight: 700; color: #475569; }
.prob-track {
    background: #f1f5f9;
    border-radius: 8px;
    height: 20px;
    width: 100%;
    overflow: hidden;
    position: relative;
}
.prob-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 6px;
}
.prob-fill-label {
    font-size: 10px;
    font-weight: 700;
    color: white;
    opacity: 0;
    white-space: nowrap;
}
.prob-fill.wide .prob-fill-label { opacity: 1; }

/* ── Distribution Grid ──────────────────────────────────────────────────────── */
.dist-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 8px;
    padding: 4px 0 8px;
}
.dist-item {
    text-align: center;
    padding: 12px 4px 10px;
    border-radius: 10px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.dist-item-emoji { font-size: 22px; margin-bottom: 4px; }
.dist-item-name  { font-size: 10px; font-weight: 600; color: #475569; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.3px; }
.dist-item-pct   { font-size: 12px; font-weight: 700; color: #1e293b; }

/* ── Footer ─────────────────────────────────────────────────────────────────── */
.footer-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
    padding: 24px 28px;
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-top: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.footer-col-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6366f1;
    margin-bottom: 10px;
}
.footer-col-content {
    font-size: 12px;
    color: #475569;
    line-height: 1.7;
}
.footer-col-content b { color: #1e293b; }
.github-btn {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #1e293b;
    color: white !important;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    text-decoration: none !important;
    margin-top: 10px;
    border: none;
    cursor: pointer;
}

/* ── Note text ───────────────────────────────────────────────────────────────── */
.note-text {
    font-size: 12px;
    color: #94a3b8;
    font-style: italic;
    padding: 6px 2px;
    font-family: 'Inter', sans-serif;
}
"""

# ── Static HTML blocks ────────────────────────────────────────────────────────

HEADER_HTML = """
<div class="app-header">
  <div class="app-title">🎙 AI Speech Emotion Recognition</div>
  <div class="app-subtitle">Multimodal Deep Learning &nbsp;·&nbsp; Final Year Project</div>
  <div class="metric-pills">
    <div class="metric-pill"><span class="metric-pill-label">Accuracy</span>51.9%</div>
    <div class="metric-pill"><span class="metric-pill-label">Weighted F1</span>0.5094</div>
    <div class="metric-pill"><span class="metric-pill-label">Dataset</span>MELD · 7 Classes</div>
    <div class="metric-pill"><span class="metric-pill-label">Model</span>RoBERTa + wav2vec2</div>
    <div class="metric-pill"><span class="metric-pill-label">Random Baseline</span>14.3%</div>
  </div>
</div>
"""

STATS_HTML = """
<div class="stats-wrap">
  <table class="stats-table">
    <thead>
      <tr>
        <th>Model</th>
        <th>Dataset</th>
        <th>Test Accuracy</th>
        <th>Weighted F1</th>
        <th>Random Baseline</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>RoBERTa-base</b> + wav2vec2-base &nbsp;(sigmoid gate fusion)</td>
        <td>MELD 7-class</td>
        <td><span class="stats-badge">51.9%</span></td>
        <td><span class="stats-badge">0.5094</span></td>
        <td>14.3%</td>
      </tr>
    </tbody>
  </table>
</div>
"""

DIST_HTML = """
<div class="dist-grid">
  <div class="dist-item"><div class="dist-item-emoji">😐</div><div class="dist-item-name">Neutral</div><div class="dist-item-pct">46.0%</div></div>
  <div class="dist-item"><div class="dist-item-emoji">😲</div><div class="dist-item-name">Surprise</div><div class="dist-item-pct">12.8%</div></div>
  <div class="dist-item"><div class="dist-item-emoji">😄</div><div class="dist-item-name">Joy</div><div class="dist-item-pct">12.5%</div></div>
  <div class="dist-item"><div class="dist-item-emoji">😠</div><div class="dist-item-name">Anger</div><div class="dist-item-pct">6.1%</div></div>
  <div class="dist-item"><div class="dist-item-emoji">😢</div><div class="dist-item-name">Sadness</div><div class="dist-item-pct">6.6%</div></div>
  <div class="dist-item"><div class="dist-item-emoji">😨</div><div class="dist-item-name">Fear</div><div class="dist-item-pct">5.6%</div></div>
  <div class="dist-item"><div class="dist-item-emoji">🤢</div><div class="dist-item-name">Disgust</div><div class="dist-item-pct">3.3%</div></div>
</div>
<div class="note-text">Class imbalance handled with inverse-frequency weights + label smoothing ε=0.10</div>
"""

FOOTER_HTML = """
<div class="footer-grid">
  <div>
    <div class="footer-col-title">Architecture</div>
    <div class="footer-col-content">
      <b>Audio:</b> wav2vec2-base (frozen, mean-pool) → 768-dim<br>
      <b>Text:</b> RoBERTa-base (top-2 layers fine-tuned) → 768-dim<br>
      <b>Fusion:</b> Sigmoid gate → MLP → 7 classes<br>
      <b>Checkpoint:</b> epoch 13 + 10 resume epochs
    </div>
  </div>
  <div>
    <div class="footer-col-title">Dataset</div>
    <div class="footer-col-content">
      <b>MELD</b> — Friends TV Show<br>
      9,989 train / 1,109 dev / 2,610 test<br>
      7 emotion classes, 3 sentiment labels<br>
      Severe class imbalance (neutral ≈ 46%)
    </div>
  </div>
  <div>
    <div class="footer-col-title">Project</div>
    <div class="footer-col-content">
      CIS Final Year Project<br>
      AI-Driven Speech Emotion Recognition<br>
      <a class="github-btn" href="https://github.com/Rohith5255/AI-Based-Human-Emotion-detection-and-Sentiemnt-Profiling" target="_blank">
        ⭐ GitHub Repository
      </a>
    </div>
  </div>
</div>
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def _tab_content(btn_fn, audio_input, has_transcript=True):
    """Not used directly — layout built inline per tab."""
    pass


with gr.Blocks(
    title="AI Speech Emotion Recognition",
    theme=gr.themes.Base(),
    css=CUSTOM_CSS,
) as demo:

    # ── Header ───────────────────────────────────────────────────────────────
    gr.HTML(HEADER_HTML)
    gr.HTML(STATS_HTML)

    with gr.Accordion("📊 MELD Emotion Class Distribution", open=False):
        gr.HTML(DIST_HTML)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Upload Audio ───────────────────────────────────────────────
        with gr.Tab("📁  Upload Audio"):
            upload_audio = gr.File(
                label="Audio file  (.wav · .mp3 · .mp4 · .m4a · .ogg · .flac)",
                file_types=[".wav", ".mp4", ".mp3", ".m4a", ".ogg", ".flac"],
            )
            upload_btn = gr.Button("⚡  Analyse", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    upload_emotion = gr.HTML(value=_EMPTY_CARD, label="Predicted Emotion")
                with gr.Column(scale=1):
                    upload_sentiment = gr.HTML(value=_EMPTY_SENTIMENT, label="Sentiment")

            upload_transcript = gr.HTML(value=_EMPTY_TRANSCRIPT, label="ASR Transcript")
            upload_chart      = gr.HTML(value=_EMPTY_CHART,      label="Emotion Probabilities")

            upload_btn.click(
                fn=predict_upload,
                inputs=[upload_audio],
                outputs=[upload_emotion, upload_sentiment, upload_transcript, upload_chart],
            )

        # ── Tab 2: Microphone ─────────────────────────────────────────────────
        with gr.Tab("🎤  Microphone"):
            mic_audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record your voice",
            )
            mic_btn = gr.Button("⚡  Analyse", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    mic_emotion = gr.HTML(value=_EMPTY_CARD, label="Predicted Emotion")
                with gr.Column(scale=1):
                    mic_sentiment = gr.HTML(value=_EMPTY_SENTIMENT, label="Sentiment")

            mic_transcript = gr.HTML(value=_EMPTY_TRANSCRIPT, label="ASR Transcript")
            mic_chart      = gr.HTML(value=_EMPTY_CHART,      label="Emotion Probabilities")

            mic_btn.click(
                fn=predict_audio,
                inputs=[mic_audio],
                outputs=[mic_emotion, mic_sentiment, mic_transcript, mic_chart],
            )

        # ── Tab 3: Text Only ──────────────────────────────────────────────────
        with gr.Tab("✏️  Type Text"):
            gr.HTML('<div class="note-text" style="padding:4px 2px 8px">No audio needed — a zero audio vector is used; text signal drives the prediction.</div>')
            text_input = gr.Textbox(
                label="Enter text",
                placeholder='e.g. "I can\'t believe you did that!"',
                lines=3,
            )
            text_btn = gr.Button("⚡  Analyse", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    text_emotion = gr.HTML(value=_EMPTY_CARD, label="Predicted Emotion")
                with gr.Column(scale=1):
                    text_sentiment = gr.HTML(value=_EMPTY_SENTIMENT, label="Sentiment")

            text_chart = gr.HTML(value=_EMPTY_CHART, label="Emotion Probabilities")

            text_btn.click(
                fn=predict_text,
                inputs=[text_input],
                outputs=[text_emotion, text_sentiment, text_chart],
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, theme=gr.themes.Base())
