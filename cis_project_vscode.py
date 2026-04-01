# -*- coding: utf-8 -*-
"""
CIS Project — Multimodal Speech Emotion Recognition (SER)
Dataset  : RAVDESS
ASR      : Wav2Vec2  (best for RAVDESS — Whisper gives identical text for all samples)
Run in   : VS Code (local machine)

Improvements over baseline:
    1. Delta + Delta-Delta MFCC (120 features instead of 40)
    2. Audio data augmentation  (7.5x more training data)
    3. Class-weighted loss      (fixes neutral class imbalance)
    4. Partial text encoder unfreeze (top 2 layers — except DeBERTa, kept frozen)
    5. Pitch + RMS energy features (122 total acoustic features)

Architecture:
    MFCC(+delta+delta2+pitch+rms) + BiLSTM + Attention
    ||  Wav2Vec2 ASR -> Text Encoder (partial unfreeze) -> CLS
    -> Attention Fusion -> Classifier
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
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    AutoTokenizer,
    AutoModel,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")

# ===============================================================
# CONFIGURE THESE PATHS
# ===============================================================

DATASET_PATH   = r"C:\Users\rohit\OneDrive\Desktop\CIS\Dataset"
MODEL_SAVE_DIR = r"C:\Users\rohit\OneDrive\Desktop\CIS\saved_models_v2"
TRAIN_CACHE    = r"C:\Users\rohit\OneDrive\Desktop\CIS\train_features_v2.pkl"
TEST_CACHE     = r"C:\Users\rohit\OneDrive\Desktop\CIS\test_features_v2.pkl"

# ===============================================================
# CONTROL PANEL
# ===============================================================

RESUME_EPOCHS = 15
RESUME_MODE   = True

# ===============================================================

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# Hyperparameters
# ===============================

TARGET_SR    = 16000
N_MFCC_BASE = 40
N_FEATURES   = 122        # 40 MFCC + 40 delta + 40 delta2 + 1 pitch + 1 rms
EPOCHS       = 10
BATCH_SIZE   = 8
LR           = 5e-5       # lower LR — text encoder partially unfrozen

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

# Original Wav2Vec2 + 40 MFCC baseline results
BASELINE_RESULTS = {
    "bert":                  0.3333,
    "roberta":               0.3854,
    "distilroberta_emotion": 0.3542,
    "deberta":               0.2674,
    "albert":                0.2639,
}

# Update best_acc after each run
COMPLETED_RESULTS = {
    "bert":                  {"description": "BERT base uncased",                   "best_acc": 0.4514},
    "roberta":               {"description": "RoBERTa base",                        "best_acc": 0.3715},
    "distilroberta_emotion": {"description": "DistilRoBERTa fine-tuned on Emotion", "best_acc": 0.4167},
    "deberta":               {"description": "DeBERTa v3 base",                     "best_acc": 0.3368},
    "albert":                {"description": "ALBERT base v2",                      "best_acc": 0.3750},
}


# ===============================
# ASR — Wav2Vec2x
# ===============================

def load_asr_model():
    print("Loading ASR model (Wav2Vec2) ...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    asr_model.eval()
    print("ASR model loaded.\n")
    return processor, asr_model

def speech_to_text(audio, processor, asr_model):
    inputs = processor(
        audio, sampling_rate=TARGET_SR,
        return_tensors="pt", padding=True
    ).input_values.to(device)
    with torch.no_grad():
        logits = asr_model(inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0].lower()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "unknown"


# ===============================
# Audio Preprocessing
# ===============================

def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=TARGET_SR)
    audio, _ = librosa.effects.trim(audio)
    return audio


# ===============================
# Improvement 1: Delta MFCC + Pitch + RMS (122 features)
# ===============================

def extract_features(audio):
    mfcc   = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC_BASE)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=TARGET_SR)
    f0 = np.nan_to_num(f0).reshape(1, -1)

    rms = librosa.feature.rms(y=audio)

    min_len = min(mfcc.shape[1], f0.shape[1], rms.shape[1])
    combined = np.concatenate([
        mfcc[:, :min_len],
        delta[:, :min_len],
        delta2[:, :min_len],
        f0[:, :min_len],
        rms[:, :min_len]
    ], axis=0)

    return combined.T  # (T, 122)


# ===============================
# Improvement 2: Audio Augmentation
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
# Feature Pre-extraction (with augmentation)
# ===============================

def preextract_features(audio_paths, labels, processor, asr_model, augment=True):
    mfcc_list, texts, labels_out = [], [], []
    total = len(audio_paths)

    for i, (path, label) in enumerate(zip(audio_paths, labels)):
        if i % 50 == 0:
            print(f"  Extracting: {i}/{total} ...")
        try:
            audio = preprocess_audio(path)
            mfcc  = extract_features(audio)
            text  = clean_text(speech_to_text(audio, processor, asr_model))

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
            print(f"  [WARN] Skipped {os.path.basename(path)}: {e}")
            mfcc_list.append(np.zeros((10, N_FEATURES)))
            texts.append("unknown")
            labels_out.append(label)

    print(f"  Done — {len(mfcc_list)} samples (original + augmented).\n")
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


# ===============================
# Improvement 3: Partial Text Encoder Unfreeze
# NOTE: DeBERTa is kept fully frozen — unfreezing causes NaN loss
# ===============================

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

        # DeBERTa v3 has disentangled attention — unfreezing causes NaN loss
        # Keep it fully frozen, only train the fusion head
        if model_key == "deberta":
            print(f"  TextEncoder ({model_key}): fully frozen (DeBERTa numerical stability).")
        else:
            # Unfreeze top 2 transformer layers + pooler
            unfrozen = 0
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in [
                    "layer.10", "layer.11",           # BERT / RoBERTa (12 layers)
                    "layer.4",  "layer.5",            # DistilRoBERTa  (6 layers)
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
# Improvement 4: Class-weighted Loss
# ===============================

def compute_class_weights(labels):
    counts  = np.bincount(labels, minlength=8).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 8
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

def run_experiment(model_key, train_dataset, test_dataset, train_labels):
    print(f"\n{'='*60}")
    print(f"  Model    : {TEXT_MODELS[model_key]['description']}")
    print(f"  Features : 122-dim  |  Augmented : Yes  |  Class weights : Yes")
    unfreeze_note = "fully frozen" if model_key == "deberta" else "top 2 layers unfrozen"
    print(f"  Text enc : {unfreeze_note}")
    print(f"{'='*60}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0
    )
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    model         = EmotionRecognitionModel(model_key).to(device)
    class_weights = compute_class_weights(train_labels)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    save_path    = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_key}.pth")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion)
        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved ({best_val_acc:.4f})")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    _, final_acc, preds, true_labels = evaluate(model, test_loader, criterion)

    print(f"\n  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(true_labels, preds, target_names=label_encoder.classes_, zero_division=0))
    print("  Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    return best_val_acc


# ===============================
# Resume Training
# ===============================

def resume_training(model_key, train_dataset, test_dataset, train_labels, extra_epochs=15):
    print(f"\n{'='*60}")
    print(f"  Resuming : {TEXT_MODELS[model_key]['description']}")
    print(f"  Adding   : {extra_epochs} more epochs")
    print(f"{'='*60}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0
    )
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    save_path     = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_key}.pth")
    model         = EmotionRecognitionModel(model_key).to(device)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  Loaded checkpoint: {save_path}")

    class_weights = compute_class_weights(train_labels)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR / 2, weight_decay=1e-4
    )
    scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc  = COMPLETED_RESULTS[model_key]["best_acc"]
    print(f"  Previous best Val Acc: {best_val_acc:.4f}\n")

    for epoch in range(1, extra_epochs + 1):
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion)
        print(f"  Epoch {epoch:2d}/{extra_epochs}  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
              f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved ({best_val_acc:.4f})")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    _, final_acc, preds, true_labels = evaluate(model, test_loader, criterion)

    print(f"\n  Best Val Accuracy after resume: {best_val_acc:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(true_labels, preds, target_names=label_encoder.classes_, zero_division=0))
    print("  Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    return best_val_acc


# ===============================
# Final Comparison Table
# ===============================

def print_comparison_table(results):
    print("\n" + "="*70)
    print("  FINAL COMPARISON: Baseline (Wav2Vec2, 40 MFCC)  vs  Improved")
    print("="*70)
    print(f"  {'Model':<42} {'Baseline':>10} {'Improved':>10} {'Change':>10}")
    print("-"*70)
    for key in TEXT_MODELS:
        desc     = TEXT_MODELS[key]["description"]
        baseline = BASELINE_RESULTS.get(key, 0)
        improved = results.get(key, 0)
        diff     = improved - baseline
        arrow    = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "≈")
        print(f"  {desc:<42} {baseline:>10.4f} {improved:>10.4f} {arrow} {diff:>+.4f}")
    print("="*70)
    if results:
        best_key  = max(results, key=lambda k: results[k])
        best_base = max(BASELINE_RESULTS, key=lambda k: BASELINE_RESULTS[k])
        overall   = results[best_key] - BASELINE_RESULTS[best_base]
        print(f"\n  Best improved model : {TEXT_MODELS[best_key]['description']}  ({results[best_key]:.4f})")
        print(f"  Best baseline model : {TEXT_MODELS[best_base]['description']}  ({BASELINE_RESULTS[best_base]:.4f})")
        print(f"  Overall improvement : {'+' if overall >= 0 else ''}{overall:.4f}  ({'+' if overall >= 0 else ''}{overall*100:.2f}pp)\n")


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    # Step 1 — Load ASR model
    processor, asr_model = load_asr_model()

    # Step 2 — Scan dataset
    audio_paths, labels = scan_ravdess(DATASET_PATH)
    labels_encoded      = label_encoder.transform(labels)

    # Step 3 — Train / test split
    train_paths, test_paths, train_labels_raw, test_labels_raw = train_test_split(
        audio_paths, labels_encoded,
        test_size=0.2, random_state=42, stratify=labels_encoded
    )
    print(f"Train: {len(train_paths)}  |  Test: {len(test_paths)}\n")

    # Step 4 — Load or create feature cache
    if os.path.exists(TRAIN_CACHE) and os.path.exists(TEST_CACHE):
        print("Loading feature cache from disk ...")
        with open(TRAIN_CACHE, "rb") as f:
            train_mfcc, train_texts, train_labels = pickle.load(f)
        with open(TEST_CACHE, "rb") as f:
            test_mfcc, test_texts, test_labels = pickle.load(f)
        print(f"Loaded {len(train_mfcc)} train + {len(test_mfcc)} test samples.\n")
    else:
        print("Pre-extracting TRAIN features (with augmentation) ...")
        print("(First run ~60-90 mins — saved to cache after)\n")
        train_mfcc, train_texts, train_labels = preextract_features(
            train_paths, train_labels_raw, processor, asr_model, augment=True
        )
        with open(TRAIN_CACHE, "wb") as f:
            pickle.dump((train_mfcc, train_texts, train_labels), f)

        print("Pre-extracting TEST features (no augmentation) ...")
        test_mfcc, test_texts, test_labels = preextract_features(
            test_paths, test_labels_raw, processor, asr_model, augment=False
        )
        with open(TEST_CACHE, "wb") as f:
            pickle.dump((test_mfcc, test_texts, test_labels), f)
        print("Features saved to cache.\n")

    print(f"Training samples after augmentation: {len(train_mfcc)}")
    print(f"Test samples: {len(test_mfcc)}\n")

    # Step 5 — Build datasets
    train_dataset = SERDataset(train_mfcc, train_texts, train_labels)
    test_dataset  = SERDataset(test_mfcc,  test_texts,  test_labels)

    # Step 6 — Run experiments
    final_results = {}

    if RESUME_MODE:
        print(f"\nRESUME MODE — adding {RESUME_EPOCHS} epochs to each model\n")
        for key in TEXT_MODELS:
            save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{key}.pth")
            if not os.path.exists(save_path):
                print(f"  No checkpoint for {key} — running fresh training first ...")
                final_results[key] = run_experiment(
                    key, train_dataset, test_dataset, train_labels
                )
                COMPLETED_RESULTS[key]["best_acc"] = final_results[key]
            else:
                final_results[key] = resume_training(
                    key, train_dataset, test_dataset, train_labels,
                    extra_epochs=RESUME_EPOCHS
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
                test_loader   = torch.utils.data.DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_fn, num_workers=0
                )
                _, acc, _, _  = evaluate(model, test_loader, criterion)
                final_results[key] = acc
            else:
                final_results[key] = run_experiment(
                    key, train_dataset, test_dataset, train_labels
                )
                COMPLETED_RESULTS[key]["best_acc"] = final_results[key]

    # Step 7 — Print final comparison
    print_comparison_table(final_results)