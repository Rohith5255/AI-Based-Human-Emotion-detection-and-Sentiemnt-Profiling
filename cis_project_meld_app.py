# -*- coding: utf-8 -*-
"""
CIS Project — Multimodal SER Inference App (Gradio)
====================================================
End product: users upload audio, record via microphone, or type text to get
  • Predicted emotion  (anger / disgust / fear / joy / neutral / sadness / surprise)
  • Sentiment label    (Positive / Negative / Neutral)
  • Per-class probability bar chart

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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
print("  EmotionModel loaded.\n")
print("All models ready.\n")


# ── ffmpeg conversion: any format → wav ──────────────────────────────────────

def convert_to_wav(input_path):
    """
    Convert any audio/video file to a temp 16kHz mono WAV using ffmpeg.
    Returns the temp WAV path. Caller is responsible for deleting it.
    If the file is already a .wav, returns the original path unchanged.
    """
    import tempfile
    import subprocess
    if input_path.lower().endswith(".wav"):
        return input_path, False   # (path, is_temp)
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
    return tmp.name, True   # (path, is_temp)


# ── Core inference ────────────────────────────────────────────────────────────

def run_inference(audio_np, text):
    """
    audio_np : np.ndarray at TARGET_SR, or None (text-only mode)
    text      : str, already cleaned — can be empty string
    Returns   : (emotion_str, sentiment_str, transcript_str, probs_dict)
    """
    # Audio embedding
    if audio_np is not None:
        audio_emb = wav2vec_extractor.embed(audio_np)      # (768,)
    else:
        audio_emb = np.zeros(WAV2VEC_DIM, dtype=np.float32)  # text-only

    # ASR if no manual text
    transcript = ""
    if audio_np is not None and not text.strip():
        transcript = asr_model.transcribe(audio_np)
        text_for_model = clean_text(transcript)
    else:
        transcript     = text.strip()
        text_for_model = clean_text(text.strip()) if text.strip() else "unknown"

    # Run emotion model
    audio_tensor = torch.tensor(audio_emb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = emotion_model(audio_tensor, [text_for_model])
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (7,)

    emotion   = EMOTIONS[int(probs.argmax())]
    sentiment = SENTIMENT_MAP[emotion]

    probs_dict = {EMOTION_EMOJI[e]: float(probs[i]) for i, e in enumerate(EMOTIONS)}

    return emotion, sentiment, transcript, probs_dict


# ── Gradio callback functions ─────────────────────────────────────────────────

def predict_upload(file_obj):
    """Tab 1 — gr.File upload. file_obj has a .name attribute with the temp path."""
    if file_obj is None:
        return "—", "—", "No file uploaded.", {}

    audio_path = file_obj.name

    # Check for macOS resource-fork / hidden files
    if os.path.basename(audio_path).startswith("._") or "._" in os.path.basename(audio_path):
        return "Error", "Error", "Please upload a valid audio file.", {}

    # Check file is readable and non-empty
    try:
        with open(audio_path, "rb") as fh:
            header = fh.read(4)
    except OSError as e:
        return "Error", "Error", f"Cannot read file: {e}", {}

    if len(header) < 4:
        return "Error", "Error", "Please upload a valid audio file.", {}

    return predict_audio(audio_path)


def predict_audio(audio_path):
    """Tab 2 (mic) and shared logic — accepts a plain file path string."""
    if audio_path is None:
        return "—", "—", "No audio provided.", {}

    wav_path, is_temp = None, False
    try:
        wav_path, is_temp = convert_to_wav(audio_path)
        audio_np = load_audio(wav_path)
    except RuntimeError as e:
        return "Error", "Error", f"Please upload a valid audio file. (ffmpeg: {e})", {}
    except Exception:
        return "Error", "Error", "Please upload a valid audio file (unsupported format or corrupt).", {}
    finally:
        if is_temp and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

    emotion, sentiment, transcript, probs = run_inference(audio_np, "")

    emotion_display   = EMOTION_EMOJI.get(emotion, emotion)
    sentiment_display = SENTIMENT_EMOJI.get(sentiment, sentiment)
    transcript_display = transcript if transcript else "(ASR returned empty)"

    return emotion_display, sentiment_display, transcript_display, probs


def predict_text(text_input):
    """Tab 3 — text-only input."""
    if not text_input or not text_input.strip():
        return "—", "—", {}

    emotion, sentiment, _, probs = run_inference(None, text_input)

    emotion_display   = EMOTION_EMOJI.get(emotion, emotion)
    sentiment_display = SENTIMENT_EMOJI.get(sentiment, sentiment)

    return emotion_display, sentiment_display, probs


# ── Gradio UI ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🎙️ AI Speech Emotion Recognition & Sentiment Profiling

**Final Year Project — CIS (AI-Driven SER)**

Upload audio, record your voice, or type text to detect emotion and sentiment.

| Model | Test Accuracy | Test Weighted F1 |
|-------|--------------|-----------------|
| RoBERTa base (wav2vec2 + text) | 50.5% | 0.5009 |

*7-class task on MELD (Friends TV dataset). Random baseline = 14.3%.*
"""

with gr.Blocks(title="Emotion & Sentiment Profiling") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── Tab 1: Upload Audio ──────────────────────────────────────────────
        with gr.Tab("📁 Upload Audio"):
            with gr.Row():
                upload_audio = gr.File(
                    label="Upload audio file (.wav / .mp4 / .mp3 / .m4a / .ogg / .flac)",
                    file_types=[".wav", ".mp4", ".mp3", ".m4a", ".ogg", ".flac"],
                )
            upload_btn = gr.Button("Analyse", variant="primary")

            with gr.Row():
                upload_emotion   = gr.Textbox(label="Predicted Emotion", interactive=False)
                upload_sentiment = gr.Textbox(label="Sentiment",         interactive=False)

            upload_transcript = gr.Textbox(
                label="Transcribed Text (ASR)", interactive=False
            )
            upload_chart = gr.Label(label="Emotion Probabilities", num_top_classes=7)

            upload_btn.click(
                fn=predict_upload,
                inputs=[upload_audio],
                outputs=[upload_emotion, upload_sentiment, upload_transcript, upload_chart],
            )

        # ── Tab 2: Microphone ────────────────────────────────────────────────
        with gr.Tab("🎤 Record Microphone"):
            mic_audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record your voice",
            )
            mic_btn = gr.Button("Analyse", variant="primary")

            with gr.Row():
                mic_emotion   = gr.Textbox(label="Predicted Emotion", interactive=False)
                mic_sentiment = gr.Textbox(label="Sentiment",         interactive=False)

            mic_transcript = gr.Textbox(
                label="Transcribed Text (ASR)", interactive=False
            )
            mic_chart = gr.Label(label="Emotion Probabilities", num_top_classes=7)

            mic_btn.click(
                fn=predict_audio,
                inputs=[mic_audio],
                outputs=[mic_emotion, mic_sentiment, mic_transcript, mic_chart],
            )

        # ── Tab 3: Text Input ────────────────────────────────────────────────
        with gr.Tab("✏️ Type Text"):
            gr.Markdown(
                "_No audio needed. The model uses a zero audio vector — "
                "text alone drives the prediction._"
            )
            text_input = gr.Textbox(
                label='Enter text (e.g. "I can\'t believe you did that!")',
                lines=3,
                placeholder="Type something here ...",
            )
            text_btn = gr.Button("Analyse", variant="primary")

            with gr.Row():
                text_emotion   = gr.Textbox(label="Predicted Emotion", interactive=False)
                text_sentiment = gr.Textbox(label="Sentiment",         interactive=False)

            text_chart = gr.Label(label="Emotion Probabilities", num_top_classes=7)

            text_btn.click(
                fn=predict_text,
                inputs=[text_input],
                outputs=[text_emotion, text_sentiment, text_chart],
            )

    gr.Markdown(
        "---\n"
        "**Architecture:** wav2vec2-base (audio features) + RoBERTa-base (text) "
        "→ Gated Multimodal Fusion → 7-class emotion classifier  \n"
        "**Trained on:** MELD dataset (9,989 utterances, Friends TV)"
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())
