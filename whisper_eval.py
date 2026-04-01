# -*- coding: utf-8 -*-
"""
CIS Project — Whisper ASR Evaluation (No Retraining)
======================================================
Swaps Wav2Vec2 ASR with OpenAI Whisper for transcription.
Loads existing saved model checkpoints and evaluates accuracy
using Whisper-generated text — no retraining required.

Compares:
    Wav2Vec2 ASR results (from training)  vs  Whisper ASR results (this script)
"""

import os
import re
import pickle
import numpy as np
import librosa
import torch
import torch.nn as nn
import warnings

from transformers import (
    AutoTokenizer,
    AutoModel,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")

# ===============================================================
# CONFIGURE THESE PATHS (same as your main script)
# ===============================================================

DATASET_PATH   = r"C:\Users\rohit\OneDrive\Desktop\CIS\Dataset"
MODEL_SAVE_DIR = r"C:\Users\rohit\OneDrive\Desktop\CIS\saved_models"
TRAIN_CACHE    = r"C:\Users\rohit\OneDrive\Desktop\CIS\train_features.pkl"

# New Whisper cache — saves time if you run this script multiple times
WHISPER_TEST_CACHE = r"C:\Users\rohit\OneDrive\Desktop\CIS\whisper_test_features.pkl"

# ===============================================================
# Whisper model size — options: tiny, base, small, medium, large
# "base" is fast and good. "small" is better but slower.
WHISPER_MODEL_SIZE = "base"
# ===============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TARGET_SR  = 16000
N_MFCC     = 40
BATCH_SIZE = 8

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)

RAVDESS_MAP = {
    "01": "neutral", "02": "calm",    "03": "happy",   "04": "sad",
    "05": "angry",   "06": "fear",    "07": "disgust",  "08": "surprise",
}

TEXT_MODELS = {
    "bert": {
        "name":        "bert-base-uncased",
        "embed_dim":   768,
        "description": "BERT base uncased",
    },
    "roberta": {
        "name":        "roberta-base",
        "embed_dim":   768,
        "description": "RoBERTa base",
    },
    "distilroberta_emotion": {
        "name":        "j-hartmann/emotion-english-distilroberta-base",
        "embed_dim":   768,
        "description": "DistilRoBERTa fine-tuned on Emotion",
    },
    "deberta": {
        "name":        "microsoft/deberta-v3-base",
        "embed_dim":   768,
        "description": "DeBERTa v3 base",
    },
    "albert": {
        "name":        "albert-base-v2",
        "embed_dim":   768,
        "description": "ALBERT base v2",
    },
}

# Previous Wav2Vec2 best accuracies for comparison
WAV2VEC2_RESULTS = {
    "bert":                  0.3333,
    "roberta":               0.3715,
    "distilroberta_emotion": 0.3507,
    "deberta":               0.2674,
    "albert":                0.2639,
}


# ===============================
# Whisper ASR
# ===============================

def load_whisper_model():
    try:
        import whisper
    except ImportError:
        print("Whisper not installed. Installing now...")
        import subprocess
        subprocess.run(["pip", "install", "openai-whisper"], check=True)
        import whisper

    print(f"Loading Whisper ({WHISPER_MODEL_SIZE}) ...")
    model = whisper.load_model(WHISPER_MODEL_SIZE).to(device)
    model.eval()
    print(f"Whisper loaded.\n")
    return model

def transcribe_whisper(audio_path, whisper_model):
    """Load audio with librosa then pass numpy array to Whisper."""
    try:
        # Load audio as numpy array — avoids Windows path issues
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # Whisper expects float32 numpy array at 16kHz
        audio = audio.astype(np.float32)
        
        result = whisper_model.transcribe(
            audio,
            language="en",
            fp16=torch.cuda.is_available(),
        )
        text = result["text"].strip().lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else "unknown"
    except Exception as e:
        print(f"  [WARN] Whisper failed on {os.path.basename(audio_path)}: {e}")
        return "unknown"


# ===============================
# Audio Preprocessing + MFCC
# ===============================

def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=TARGET_SR)
    audio, _ = librosa.effects.trim(audio)
    return audio

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)
    return mfcc.T


# ===============================
# Model Architecture (same as training)
# ===============================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return torch.sum(weights * lstm_out, dim=1)


class AcousticEncoder(nn.Module):
    def __init__(self, input_dim=N_MFCC, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            batch_first=True, bidirectional=True,
            num_layers=2, dropout=0.3,
        )
        self.attention = TemporalAttention(hidden_dim)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        return self.attention(out)


class TextEncoder(nn.Module):
    def __init__(self, model_key):
        super().__init__()
        cfg            = TEXT_MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
        self.model     = AutoModel.from_pretrained(cfg["name"], use_safetensors=True)
        self.embed_dim = cfg["embed_dim"]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]


class MultimodalFusion(nn.Module):
    def __init__(self, acoustic_dim=256, semantic_dim=768, hidden_dim=256, num_classes=8):
        super().__init__()
        fusion_dim      = acoustic_dim + semantic_dim
        self.attention  = nn.Linear(fusion_dim, fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, acoustic_vec, semantic_vec):
        fused  = torch.cat((acoustic_vec, semantic_vec), dim=1)
        attn_w = torch.softmax(self.attention(fused), dim=1)
        return self.classifier(fused * attn_w)


class EmotionRecognitionModel(nn.Module):
    def __init__(self, model_key):
        super().__init__()
        self.acoustic_encoder = AcousticEncoder()
        self.text_encoder     = TextEncoder(model_key)
        self.fusion           = MultimodalFusion(semantic_dim=self.text_encoder.embed_dim)

    def forward(self, mfcc_padded, lengths, texts):
        a = self.acoustic_encoder(mfcc_padded, lengths)
        s = self.text_encoder(texts)
        return self.fusion(a, s)


# ===============================
# Dataset + Collate
# ===============================

class SERDataset(torch.utils.data.Dataset):
    def __init__(self, mfcc_list, texts, labels):
        self.mfcc_list = mfcc_list
        self.texts     = texts
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mfcc_list[idx], self.texts[idx], self.labels[idx]


def collate_fn(batch):
    mfccs, texts, lbls = zip(*batch)
    tensors = [torch.tensor(m, dtype=torch.float32) for m in mfccs]
    lengths = torch.tensor([t.shape[0] for t in tensors])
    padded  = pad_sequence(tensors, batch_first=True)
    labels  = torch.tensor(lbls, dtype=torch.long)
    return padded, lengths, list(texts), labels


# ===============================
# Dataset Scanning
# ===============================

def scan_ravdess(dataset_path):
    audio_paths, labels = [], []
    for actor in sorted(os.listdir(dataset_path)):
        actor_folder = os.path.join(dataset_path, actor)
        if not os.path.isdir(actor_folder):
            continue
        for fname in os.listdir(actor_folder):
            if not fname.endswith(".wav"):
                continue
            parts = fname.split("-")
            if len(parts) < 3 or parts[2] not in RAVDESS_MAP:
                continue
            audio_paths.append(os.path.join(actor_folder, fname))
            labels.append(RAVDESS_MAP[parts[2]])
    print(f"Total samples found: {len(audio_paths)}")
    return audio_paths, labels


# ===============================
# Evaluation with saved checkpoint
# ===============================

def evaluate_model(model_key, test_mfcc, whisper_texts, test_labels):
    """Load saved checkpoint and evaluate with Whisper transcriptions."""
    print(f"\n{'='*60}")
    print(f"  Evaluating : {TEXT_MODELS[model_key]['description']}")
    print(f"  ASR        : Whisper ({WHISPER_MODEL_SIZE})")
    print(f"{'='*60}")

    save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_key}.pth")
    if not os.path.exists(save_path):
        print(f"  [SKIP] No checkpoint found at {save_path}")
        return None

    model = EmotionRecognitionModel(model_key).to(device)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    print(f"  Loaded checkpoint: {save_path}")

    dataset    = SERDataset(test_mfcc, whisper_texts, test_labels)
    loader     = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for mfcc_padded, lengths, texts, labels in loader:
            mfcc_padded = mfcc_padded.to(device)
            logits      = model(mfcc_padded, lengths, texts)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)

    print(f"\n  Whisper Val Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Wav2Vec2 Val Accuracy: {WAV2VEC2_RESULTS[model_key]:.4f}  ({WAV2VEC2_RESULTS[model_key]*100:.2f}%)")
    diff = acc - WAV2VEC2_RESULTS[model_key]
    direction = "+" if diff >= 0 else ""
    print(f"  Change               : {direction}{diff:.4f}  ({direction}{diff*100:.2f}pp)")

    print(f"\n  Classification Report (Whisper):\n")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_, zero_division=0
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return acc


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    # Step 1 — Load Whisper
    whisper_model = load_whisper_model()

    # Step 2 — Scan dataset and get the same test split
    audio_paths, labels = scan_ravdess(DATASET_PATH)
    labels_encoded      = label_encoder.transform(labels)

    _, test_paths, _, test_labels = train_test_split(
        audio_paths, labels_encoded,
        test_size=0.2, random_state=42, stratify=labels_encoded
    )
    print(f"Test samples: {len(test_paths)}\n")

    # Step 3 — Load MFCC from existing cache (no re-extraction needed)
    print("Loading MFCC cache from disk ...")
    with open(TRAIN_CACHE, "rb") as f:
        pass  # just check it exists

    # We need test MFCCs — load from the test cache
    TEST_CACHE = r"C:\Users\rohit\OneDrive\Desktop\CIS\test_features.pkl"
    with open(TEST_CACHE, "rb") as f:
        test_mfcc, _ = pickle.load(f)   # discard old Wav2Vec2 texts
    print(f"Loaded {len(test_mfcc)} test MFCC features from cache.\n")

    # Step 4 — Transcribe test set with Whisper (or load from Whisper cache)
    if os.path.exists(WHISPER_TEST_CACHE):
        print("Loading Whisper transcription cache ...")
        with open(WHISPER_TEST_CACHE, "rb") as f:
            whisper_texts = pickle.load(f)
        print(f"Loaded {len(whisper_texts)} Whisper transcriptions from cache.\n")
    else:
        print(f"Transcribing {len(test_paths)} test files with Whisper ({WHISPER_MODEL_SIZE}) ...")
        print("(This takes ~5-15 minutes depending on GPU speed)\n")
        whisper_texts = []
        for i, path in enumerate(test_paths):
            if i % 30 == 0:
                print(f"  Transcribing: {i}/{len(test_paths)} ...")
            text = transcribe_whisper(path, whisper_model)
            whisper_texts.append(text)
        with open(WHISPER_TEST_CACHE, "wb") as f:
            pickle.dump(whisper_texts, f)
        print(f"\n  Done. Whisper transcriptions saved to cache.\n")

    # Show a few sample transcriptions for verification
    print("Sample Whisper transcriptions (first 5 test files):")
    for i in range(min(5, len(whisper_texts))):
        print(f"  [{i+1}] {os.path.basename(test_paths[i])}  →  '{whisper_texts[i]}'")
    print()

    # Step 5 — Evaluate all 5 models with Whisper transcriptions
    whisper_results = {}
    for key in TEXT_MODELS:
        acc = evaluate_model(key, test_mfcc, whisper_texts, test_labels)
        if acc is not None:
            whisper_results[key] = acc

    # Step 6 — Final comparison table
    print("\n" + "="*70)
    print("  FINAL COMPARISON: Wav2Vec2 ASR  vs  Whisper ASR")
    print("="*70)
    print(f"  {'Model':<42} {'Wav2Vec2':>10} {'Whisper':>10} {'Change':>10}")
    print("-"*70)
    for key in TEXT_MODELS:
        desc  = TEXT_MODELS[key]["description"]
        w2v   = WAV2VEC2_RESULTS.get(key, 0)
        whi   = whisper_results.get(key, 0)
        diff  = whi - w2v
        arrow = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "≈")
        print(f"  {desc:<42} {w2v:>10.4f} {whi:>10.4f} {arrow} {diff:+.4f}")
    print("="*70)

    best_w2v = max(WAV2VEC2_RESULTS, key=lambda k: WAV2VEC2_RESULTS[k])
    if whisper_results:
        best_whi = max(whisper_results, key=lambda k: whisper_results[k])
        print(f"\n  Wav2Vec2 winner : {TEXT_MODELS[best_w2v]['description']}  ({WAV2VEC2_RESULTS[best_w2v]:.4f})")
        print(f"  Whisper winner  : {TEXT_MODELS[best_whi]['description']}  ({whisper_results[best_whi]:.4f})")
        overall_diff = whisper_results[best_whi] - WAV2VEC2_RESULTS[best_w2v]
        print(f"  Overall change  : {'+' if overall_diff >= 0 else ''}{overall_diff:.4f} ({'+' if overall_diff >= 0 else ''}{overall_diff*100:.2f}pp)\n")
