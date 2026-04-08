# -*- coding: utf-8 -*-
"""
CIS Project — Multimodal Speech Emotion Recognition (SER)
Dataset  : MELD (Multimodal EmotionLines Dataset — Friends TV Series)
Text     : Gold transcriptions from CSV (no ASR needed — text varies per utterance)
Audio    : Pre-extracted WAV files at 16kHz

Why MELD over RAVDESS:
    RAVDESS uses only 2 scripted sentences repeated across all emotions, so ASR
    transcriptions are nearly identical regardless of emotion — the text encoder
    contributes almost zero signal. MELD has 13,000+ natural conversational
    utterances where the text content genuinely differs per emotion, allowing the
    text encoders to contribute real discriminative signal.

Key differences from RAVDESS script:
    1. No ASR model needed — MELD CSV has gold transcriptions
    2. Pre-split train/dev/test sets (use official splits, no random split)
    3. 7 emotion classes (anger, disgust, sadness, joy, neutral, surprise, fear)
    4. Audio filenames are indexed by dialogue_id + utterance_id (not RAVDESS code)
    5. Class imbalance is severe (Neutral ~46%) — class weights are critical
    6. Audio quality varies (TV recording) — augmentation helps generalization

Architecture (same as RAVDESS version):
    MFCC(+delta+delta2+pitch+rms) + BiLSTM + Attention
    ||  Gold text transcript -> Text Encoder (partial unfreeze) -> CLS
    -> Attention Fusion -> Classifier

Download MELD:
    wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
    tar -xzf MELD.Raw.tar.gz
    This gives you: MELD.Raw/train_splits/, MELD.Raw/dev_splits_complete/,
                    MELD.Raw/output_repeated_splits_test/,
                    MELD.Raw/train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import warnings

from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")

# Ensure ffmpeg (installed via winget) is on PATH for librosa mp4 decoding
_ffmpeg_bin = r"C:\Users\rohit\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
if _ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

# ===============================================================
# CONFIGURE THESE PATHS
# ===============================================================

# Root folder after extracting MELD.Raw.tar.gz
MELD_ROOT = r"C:\Users\rohit\MELD.Raw\MELD.Raw"

# Where audio splits live (subfolders of MELD_ROOT)
TRAIN_AUDIO_DIR = os.path.join(MELD_ROOT, "train_splits")
DEV_AUDIO_DIR   = os.path.join(MELD_ROOT, "dev_splits_complete")
TEST_AUDIO_DIR  = os.path.join(MELD_ROOT, "output_repeated_splits_test")

# CSV annotation files (also inside MELD_ROOT)
TRAIN_CSV = os.path.join(MELD_ROOT, "train_sent_emo.csv")
DEV_CSV   = os.path.join(MELD_ROOT, "dev_sent_emo.csv")
TEST_CSV  = os.path.join(MELD_ROOT, "test_sent_emo.csv")

# Output dirs
MODEL_SAVE_DIR = r"D:\SEAD\meld_cache\saved_models_meld"
TRAIN_CACHE    = r"D:\SEAD\meld_cache\meld_train_features.pkl"
DEV_CACHE      = r"D:\SEAD\meld_cache\meld_dev_features.pkl"
TEST_CACHE     = r"D:\SEAD\meld_cache\meld_test_features.pkl"

# ===============================================================
# CONTROL PANEL
# ===============================================================

RESUME_MODE   = False   # Set True to resume from existing checkpoints
RESUME_EPOCHS = 10      # Extra epochs when resuming

# ===============================================================

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# Hyperparameters
# ===============================

TARGET_SR    = 16000
N_MFCC_BASE  = 40
N_FEATURES   = 122        # 40 MFCC + 40 delta + 40 delta2 + 1 pitch + 1 rms
EPOCHS       = 12
BATCH_SIZE   = 16         # Larger batch — MELD is bigger than RAVDESS
LR           = 3e-5

# MELD 7 emotion classes (lowercase to match CSV after .lower())
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)

# Map CSV Emotion column values to our lowercase labels
MELD_EMOTION_MAP = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "joy",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": "surprise",
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


# ===============================
# Audio Preprocessing
# ===============================

def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    # MELD clips are short (avg ~2s) — pad if < 0.5s to avoid librosa errors
    if len(audio) < TARGET_SR // 2:
        audio = np.pad(audio, (0, TARGET_SR // 2 - len(audio)))
    return audio


# ===============================
# Feature Extraction (122-dim)
# ===============================

def extract_features(audio):
    mfcc   = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC_BASE)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    f0 = librosa.yin(audio, fmin=50, fmax=400, sr=TARGET_SR)
    f0 = np.nan_to_num(f0).reshape(1, -1)

    rms = librosa.feature.rms(y=audio)

    min_len = min(mfcc.shape[1], f0.shape[1], rms.shape[1])
    combined = np.concatenate([
        mfcc[:, :min_len],
        delta[:, :min_len],
        delta2[:, :min_len],
        f0[:, :min_len],
        rms[:, :min_len],
    ], axis=0)

    return combined.T  # (T, 122)


# ===============================
# Audio Augmentation
# ===============================

def augment_audio(audio, sr=TARGET_SR):
    augmented = []
    try:
        augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
        augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    except Exception:
        pass
    try:
        augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
        augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))
    except Exception:
        pass
    noise = np.random.normal(0, 0.004, len(audio)).astype(np.float32)
    augmented.append(audio + noise)
    return augmented


# ===============================
# MELD Audio Filename Resolution
# ===============================

def build_audio_path(audio_dir, dialogue_id, utterance_id):
    """
    MELD audio files are named: diaX_uttY.mp4 (or .wav after extraction).
    Some Kaggle versions extract as wav directly; official tar gives mp4.
    We try wav first, then mp4 (ffmpeg can read mp4 audio via librosa).
    """
    base = f"dia{dialogue_id}_utt{utterance_id}"
    for ext in [".wav", ".mp4"]:
        path = os.path.join(audio_dir, base + ext)
        if os.path.exists(path):
            return path
    return None  # missing file — will be skipped


# ===============================
# Dataset Loading from CSV
# ===============================

def load_meld_csv(csv_path):
    """Load MELD CSV and return list of (utterance_text, emotion_label, dialogue_id, utterance_id)."""
    df = pd.read_csv(csv_path)
    # Normalise column names — some versions use slightly different casing
    df.columns = [c.strip() for c in df.columns]

    records = []
    for _, row in df.iterrows():
        emotion_raw = str(row["Emotion"]).strip().lower()
        emotion = MELD_EMOTION_MAP.get(emotion_raw)
        if emotion is None:
            continue  # skip unknown labels
        utterance = str(row["Utterance"]).strip()
        dialogue_id  = int(row["Dialogue_ID"])
        utterance_id = int(row["Utterance_ID"])
        records.append((utterance, emotion, dialogue_id, utterance_id))
    return records


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s']", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "unknown"


# ===============================
# Feature Pre-extraction
# ===============================

def preextract_features(records, audio_dir, augment=False):
    """
    Extract MFCC features + use gold text from CSV (no ASR needed).
    Returns: mfcc_list, texts, labels (as encoded ints)
    """
    mfcc_list, texts, labels_out = [], [], []
    skipped = 0
    total = len(records)

    for i, (utterance, emotion, dia_id, utt_id) in enumerate(records):
        if i % 200 == 0:
            print(f"  Extracting: {i}/{total}  (skipped: {skipped}) ...")

        audio_path = build_audio_path(audio_dir, dia_id, utt_id)
        if audio_path is None:
            skipped += 1
            continue

        try:
            audio = preprocess_audio(audio_path)
            mfcc  = extract_features(audio)
            text  = clean_text(utterance)
            label = label_encoder.transform([emotion])[0]

            mfcc_list.append(mfcc)
            texts.append(text)
            labels_out.append(label)

            if augment:
                for aug_audio in augment_audio(audio):
                    try:
                        aug_mfcc = extract_features(aug_audio)
                        mfcc_list.append(aug_mfcc)
                        texts.append(text)
                        labels_out.append(label)
                    except Exception:
                        pass

        except Exception as e:
            skipped += 1
            if i < 20:  # only print first few warnings
                print(f"  [WARN] Skipped dia{dia_id}_utt{utt_id}: {e}")

    print(f"  Done — {len(mfcc_list)} samples ({skipped} skipped).\n")
    return mfcc_list, texts, np.array(labels_out)


# ===============================
# Model Architecture
# ===============================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return torch.sum(weights * lstm_out, dim=1)


class AcousticEncoder(nn.Module):
    def __init__(self, input_dim=N_FEATURES, hidden_dim=128):
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
        self.model_key = model_key

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # DeBERTa kept fully frozen for numerical stability
        if model_key == "deberta":
            print(f"  TextEncoder ({model_key}): fully frozen.")
        else:
            unfrozen = 0
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in [
                    "layer.10", "layer.11",
                    "layer.4",  "layer.5",
                    "encoder.layer.10", "encoder.layer.11",
                    "pooler",
                ]):
                    param.requires_grad = True
                    unfrozen += 1
            print(f"  TextEncoder ({model_key}): {unfrozen} parameter groups unfrozen.")

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]


class MultimodalFusion(nn.Module):
    def __init__(self, acoustic_dim=256, semantic_dim=768, hidden_dim=256, num_classes=7):
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
        self.fusion           = MultimodalFusion(
            semantic_dim=self.text_encoder.embed_dim,
            num_classes=len(EMOTIONS),
        )

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
# Class-weighted Loss
# ===============================

def compute_class_weights(labels, num_classes=7):
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ===============================
# Training Loop
# ===============================

def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for mfcc_padded, lengths, texts, labels in loader:
        mfcc_padded = mfcc_padded.to(device)
        labels      = labels.to(device)
        optimizer.zero_grad()
        logits      = model(mfcc_padded, lengths, texts)
        loss        = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    if scheduler:
        scheduler.step()
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


# ===============================
# Evaluation Loop
# ===============================

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for mfcc_padded, lengths, texts, labels in loader:
            mfcc_padded = mfcc_padded.to(device)
            labels      = labels.to(device)
            logits      = model(mfcc_padded, lengths, texts)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), all_preds, all_labels


# ===============================
# Fresh Training
# ===============================

def run_experiment(model_key, train_dataset, dev_dataset, test_dataset, train_labels):
    print(f"\n{'='*60}")
    print(f"  Model    : {TEXT_MODELS[model_key]['description']}")
    print(f"  Dataset  : MELD (gold transcriptions, no ASR)")
    print(f"  Features : 122-dim  |  Augmented : Yes  |  Class weights : Yes")
    unfreeze_note = "fully frozen" if model_key == "deberta" else "top 2 layers unfrozen"
    print(f"  Text enc : {unfreeze_note}")
    print(f"{'='*60}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    model         = EmotionRecognitionModel(model_key).to(device)
    class_weights = compute_class_weights(train_labels)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    save_path    = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_key}.pth")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        vl_loss, vl_acc, _, _ = evaluate(model, dev_loader, criterion)
        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved ({best_val_acc:.4f})")

    # Final evaluation on held-out test set
    model.load_state_dict(torch.load(save_path, weights_only=True))
    _, test_acc, preds, true_labels = evaluate(model, test_loader, criterion)

    print(f"\n  Best Dev Accuracy : {best_val_acc:.4f}")
    print(f"  Test Accuracy     : {test_acc:.4f}")
    print(f"\n  Classification Report (Test):\n")
    print(classification_report(
        true_labels, preds,
        target_names=label_encoder.classes_, zero_division=0,
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    return best_val_acc, test_acc


# ===============================
# Resume Training
# ===============================

def resume_training(model_key, train_dataset, dev_dataset, test_dataset,
                    train_labels, previous_best_acc, extra_epochs=10):
    print(f"\n{'='*60}")
    print(f"  Resuming : {TEXT_MODELS[model_key]['description']}")
    print(f"  Adding   : {extra_epochs} more epochs")
    print(f"{'='*60}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_key}.pth")
    model     = EmotionRecognitionModel(model_key).to(device)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  Loaded checkpoint: {save_path}")

    class_weights = compute_class_weights(train_labels)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR / 3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=extra_epochs)

    best_val_acc = previous_best_acc
    print(f"  Previous best Dev Acc: {best_val_acc:.4f}\n")

    for epoch in range(1, extra_epochs + 1):
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        vl_loss, vl_acc, _, _ = evaluate(model, dev_loader, criterion)
        print(f"  Epoch {epoch:2d}/{extra_epochs}  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved ({best_val_acc:.4f})")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    _, test_acc, preds, true_labels = evaluate(model, test_loader, criterion)

    print(f"\n  Best Dev Acc after resume : {best_val_acc:.4f}")
    print(f"  Test Accuracy             : {test_acc:.4f}")
    print(f"\n  Classification Report (Test):\n")
    print(classification_report(
        true_labels, preds,
        target_names=label_encoder.classes_, zero_division=0,
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    return best_val_acc, test_acc


# ===============================
# Final Comparison Table
# ===============================

def print_results_table(dev_results, test_results):
    print("\n" + "="*65)
    print("  MELD RESULTS — Multimodal SER with Gold Transcriptions")
    print("="*65)
    print(f"  {'Model':<42} {'Dev Acc':>10} {'Test Acc':>10}")
    print("-"*65)
    for key in TEXT_MODELS:
        desc = TEXT_MODELS[key]["description"]
        dev  = dev_results.get(key, 0)
        test = test_results.get(key, 0)
        print(f"  {desc:<42} {dev:>10.4f} {test:>10.4f}")
    print("="*65)
    if test_results:
        best_key = max(test_results, key=lambda k: test_results[k])
        print(f"\n  Best model (test): {TEXT_MODELS[best_key]['description']}  "
              f"({test_results[best_key]:.4f})\n")


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    # Step 1 — Load CSV records
    print("Loading MELD CSV annotations ...")
    train_records = load_meld_csv(TRAIN_CSV)
    dev_records   = load_meld_csv(DEV_CSV)
    test_records  = load_meld_csv(TEST_CSV)
    print(f"  Train: {len(train_records)}  Dev: {len(dev_records)}  Test: {len(test_records)}\n")

    # Step 2 — Load or create feature caches
    if os.path.exists(TRAIN_CACHE) and os.path.exists(DEV_CACHE) and os.path.exists(TEST_CACHE):
        print("Loading feature caches from disk ...")
        with open(TRAIN_CACHE, "rb") as f:
            train_mfcc, train_texts, train_labels = pickle.load(f)
        with open(DEV_CACHE, "rb") as f:
            dev_mfcc, dev_texts, dev_labels = pickle.load(f)
        with open(TEST_CACHE, "rb") as f:
            test_mfcc, test_texts, test_labels = pickle.load(f)
        print(f"  Loaded {len(train_mfcc)} train + {len(dev_mfcc)} dev + {len(test_mfcc)} test.\n")
    else:
        print("Pre-extracting TRAIN features (with augmentation) ...")
        train_mfcc, train_texts, train_labels = preextract_features(
            train_records, TRAIN_AUDIO_DIR, augment=True
        )
        with open(TRAIN_CACHE, "wb") as f:
            pickle.dump((train_mfcc, train_texts, train_labels), f)

        print("Pre-extracting DEV features ...")
        dev_mfcc, dev_texts, dev_labels = preextract_features(
            dev_records, DEV_AUDIO_DIR, augment=False
        )
        with open(DEV_CACHE, "wb") as f:
            pickle.dump((dev_mfcc, dev_texts, dev_labels), f)

        print("Pre-extracting TEST features ...")
        test_mfcc, test_texts, test_labels = preextract_features(
            test_records, TEST_AUDIO_DIR, augment=False
        )
        with open(TEST_CACHE, "wb") as f:
            pickle.dump((test_mfcc, test_texts, test_labels), f)
        print("Feature caches saved.\n")

    print(f"Training samples (after augmentation): {len(train_mfcc)}")
    print(f"Dev samples: {len(dev_mfcc)}  |  Test samples: {len(test_mfcc)}\n")

    # Class distribution (useful to verify imbalance)
    unique, counts = np.unique(train_labels, return_counts=True)
    print("Train label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {label_encoder.classes_[u]:<12}: {c:5d}  ({100*c/len(train_labels):.1f}%)")
    print()

    # Step 3 — Build datasets
    train_dataset = SERDataset(train_mfcc, train_texts, train_labels)
    dev_dataset   = SERDataset(dev_mfcc,   dev_texts,   dev_labels)
    test_dataset  = SERDataset(test_mfcc,  test_texts,  test_labels)

    # Step 4 — Run experiments
    dev_results  = {}
    test_results = {}

    if RESUME_MODE:
        print(f"\nRESUME MODE — adding {RESUME_EPOCHS} epochs\n")
        # You must manually populate previous_best from your earlier run
        previous_bests = {k: 0.0 for k in TEXT_MODELS}
        for key in TEXT_MODELS:
            save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{key}.pth")
            if not os.path.exists(save_path):
                print(f"  No checkpoint for {key} — running fresh training ...")
                dev_results[key], test_results[key] = run_experiment(
                    key, train_dataset, dev_dataset, test_dataset, train_labels
                )
            else:
                dev_results[key], test_results[key] = resume_training(
                    key, train_dataset, dev_dataset, test_dataset,
                    train_labels, previous_bests[key], RESUME_EPOCHS
                )
    else:
        print("\nFRESH MODE — training all models from scratch\n")
        for key in TEXT_MODELS:
            save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{key}.pth")
            if os.path.exists(save_path):
                print(f"  Checkpoint exists for {key} — skipping (delete to retrain).")
                model         = EmotionRecognitionModel(key).to(device)
                model.load_state_dict(torch.load(save_path, weights_only=True))
                class_weights = compute_class_weights(train_labels)
                criterion     = nn.CrossEntropyLoss(weight=class_weights)
                dev_loader    = torch.utils.data.DataLoader(
                    dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_fn, num_workers=0,
                )
                test_loader   = torch.utils.data.DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_fn, num_workers=0,
                )
                _, dev_acc, _, _   = evaluate(model, dev_loader, criterion)
                _, test_acc, _, _  = evaluate(model, test_loader, criterion)
                dev_results[key]   = dev_acc
                test_results[key]  = test_acc
            else:
                dev_results[key], test_results[key] = run_experiment(
                    key, train_dataset, dev_dataset, test_dataset, train_labels
                )

    # Step 5 — Print final table
    print_results_table(dev_results, test_results)
