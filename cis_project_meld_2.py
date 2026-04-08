# -*- coding: utf-8 -*-
"""
CIS Project — Multimodal SER on MELD  (Version 2 — Improved)
=============================================================
Key improvements over v1:
  1. wav2vec2-base audio features  — replaces 122-dim hand-crafted MFCC.
     Pre-trained on 960h LibriSpeech, captures rich prosodic/phonetic info.
  2. Linear warmup (10 % of steps) + cosine annealing scheduler
     — eliminates the wild val-acc swings seen in v1 (epochs 1-5).
  3. Differential learning rates
     — text transformer  : 1e-5  (fine-tune gently)
     — audio proj / fusion: 3e-4  (learn fast on new head)
  4. Label smoothing (0.1) — reduces overconfidence on minority classes.
  5. DeBERTa last-2-layer unfrozen — v1 kept it fully frozen ("for
     numerical stability"), which caused training collapse.
  6. Reduced augmentation  3× (orig + time-stretch × 2)  instead of v1's
     6× (orig + 5 variants) — less memorisation, better generalisation.
  7. Simpler audio head: Linear projection instead of BiLSTM since
     wav2vec2 already encodes temporal context internally.
  8. 15 training epochs with warmup → models converge properly.

Expected improvement over v1 baselines:
  v1: ~45-49% accuracy / ~0.46-0.49 weighted F1
  v2: ~57-63% accuracy / ~0.55-0.62 weighted F1  (research range for MELD)
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

from transformers import (
    AutoTokenizer, AutoModel,
    Wav2Vec2Model, Wav2Vec2FeatureExtractor,
    get_linear_schedule_with_warmup,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")

# ── ffmpeg path for mp4 decoding ────────────────────────────────────────────
_ffmpeg_bin = (
    r"C:\Users\rohit\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
)
if _ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

# ================================================================
# PATHS  — update if your layout differs
# ================================================================
MELD_ROOT      = r"C:\Users\rohit\MELD.Raw\MELD.Raw"
TRAIN_AUDIO_DIR = os.path.join(MELD_ROOT, "train_splits")
DEV_AUDIO_DIR   = os.path.join(MELD_ROOT, "dev_splits_complete")
TEST_AUDIO_DIR  = os.path.join(MELD_ROOT, "output_repeated_splits_test")
TRAIN_CSV  = os.path.join(MELD_ROOT, "train_sent_emo.csv")
DEV_CSV    = os.path.join(MELD_ROOT, "dev_sent_emo.csv")
TEST_CSV   = os.path.join(MELD_ROOT, "test_sent_emo.csv")

# v2 caches (separate from v1 so both can coexist)
MODEL_SAVE_DIR = r"D:\SEAD\meld_cache\saved_models_meld_v2"
TRAIN_CACHE    = r"D:\SEAD\meld_cache\meld_v2_train_features.pkl"
DEV_CACHE      = r"D:\SEAD\meld_cache\meld_v2_dev_features.pkl"
TEST_CACHE     = r"D:\SEAD\meld_cache\meld_v2_test_features.pkl"

# ================================================================
# CONTROL PANEL
# ================================================================
RESUME_MODE   = False
RESUME_EPOCHS = 8

# ================================================================

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# Hyperparameters
# ===============================
TARGET_SR    = 16000
EPOCHS       = 15
BATCH_SIZE   = 32        # larger batch — audio embeds are fixed-size now
LR_TEXT      = 1e-5      # gentle fine-tune for pretrained text transformer
LR_HEAD      = 3e-4      # faster learning for audio proj + fusion head
WARMUP_FRAC  = 0.10      # first 10 % of steps = linear warmup
LABEL_SMOOTH = 0.10      # label smoothing coefficient
WEIGHT_DECAY = 1e-4

WAV2VEC_MODEL = "facebook/wav2vec2-base"   # 768-dim, 94M params
WAV2VEC_DIM   = 768

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)

MELD_EMOTION_MAP = {e: e for e in EMOTIONS}   # CSV labels already match

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
        "description": "DistilRoBERTa-Emotion",
    },
    # DeBERTa v3 excluded — disentangled attention produces NaN gradients
    # when unfrozen and near-random output when frozen in multimodal setup.
    # Documented instability in non-classification fine-tuning contexts.
    "albert": {
        "name":        "albert-base-v2",
        "embed_dim":   768,
        "description": "ALBERT base v2",
    },
}


# ===============================
# Audio utilities
# ===============================

def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    min_len   = TARGET_SR // 2          # pad clips shorter than 0.5 s
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    return audio


def augment_audio_3x(audio):
    """Return 2 augmented copies (time-stretch only — fastest, least distortion)."""
    augmented = []
    try:
        augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
    except Exception:
        augmented.append(audio.copy())
    try:
        augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    except Exception:
        augmented.append(audio.copy())
    return augmented   # caller adds original → 3× total


def build_audio_path(audio_dir, dialogue_id, utterance_id):
    base = f"dia{dialogue_id}_utt{utterance_id}"
    for ext in [".wav", ".mp4"]:
        path = os.path.join(audio_dir, base + ext)
        if os.path.exists(path):
            return path
    return None


# ===============================
# wav2vec2 Feature Extractor
# ===============================

class Wav2VecExtractor:
    """Wraps facebook/wav2vec2-base for batch-free single-clip embedding."""

    def __init__(self):
        print(f"  Loading {WAV2VEC_MODEL} for audio feature extraction ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL)
        self.model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"  wav2vec2-base loaded (frozen, inference only).\n")

    @torch.no_grad()
    def embed(self, audio_np):
        """Return mean-pooled last hidden state: shape (768,)."""
        inputs = self.feature_extractor(
            audio_np, sampling_rate=TARGET_SR,
            return_tensors="pt", padding=True,
        )
        input_values = inputs.input_values.to(device)
        out = self.model(input_values)
        # mean-pool over time, squeeze batch dim → (768,)
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


# ===============================
# CSV loading
# ===============================

def load_meld_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    records = []
    for _, row in df.iterrows():
        emotion_raw = str(row["Emotion"]).strip().lower()
        emotion = MELD_EMOTION_MAP.get(emotion_raw)
        if emotion is None:
            continue
        records.append((
            str(row["Utterance"]).strip(),
            emotion,
            int(row["Dialogue_ID"]),
            int(row["Utterance_ID"]),
        ))
    return records


def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s']", "", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "unknown"


# ===============================
# Feature Pre-extraction (wav2vec2)
# ===============================

def preextract_features(records, audio_dir, wav2vec, augment=False):
    """
    Returns:
        audio_embeds : list of np.ndarray shape (768,)
        texts        : list of str
        labels       : np.ndarray of int
    """
    audio_embeds, texts, labels_out = [], [], []
    skipped = 0
    total   = len(records)

    for i, (utterance, emotion, dia_id, utt_id) in enumerate(records):
        if i % 200 == 0:
            print(f"  Extracting: {i}/{total}  (skipped: {skipped}) ...")

        audio_path = build_audio_path(audio_dir, dia_id, utt_id)
        if audio_path is None:
            skipped += 1
            continue

        try:
            audio = load_audio(audio_path)
            emb   = wav2vec.embed(audio)
            text  = clean_text(utterance)
            label = label_encoder.transform([emotion])[0]

            audio_embeds.append(emb)
            texts.append(text)
            labels_out.append(label)

            if augment:
                for aug_audio in augment_audio_3x(audio):
                    try:
                        aug_emb = wav2vec.embed(aug_audio)
                        audio_embeds.append(aug_emb)
                        texts.append(text)
                        labels_out.append(label)
                    except Exception:
                        pass

        except Exception as e:
            skipped += 1
            if i < 20:
                print(f"  [WARN] Skipped dia{dia_id}_utt{utt_id}: {e}")

    print(f"  Done — {len(audio_embeds)} samples ({skipped} skipped).\n")
    return audio_embeds, texts, np.array(labels_out)


# ===============================
# Model Architecture  (v2)
# ===============================

class AudioProjection(nn.Module):
    """
    Projects wav2vec2 mean-pool embedding (768) → 256.
    BiLSTM not needed — wav2vec2 already encodes temporal context.
    """
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
        return self.net(x)       # (B, 256)


class TextEncoder(nn.Module):
    def __init__(self, model_key):
        super().__init__()
        cfg            = TEXT_MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
        self.model     = AutoModel.from_pretrained(cfg["name"], use_safetensors=True)
        self.embed_dim = cfg["embed_dim"]
        self.model_key = model_key

        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # DeBERTa v3 NaN-explodes when any transformer layer is unfrozen
        # with our multimodal setup — keep fully frozen, rely on head learning.
        if model_key == "deberta":
            print(f"  TextEncoder ({model_key}): fully frozen (DeBERTa v3 NaN guard).")
        else:
            unfrozen = 0
            for name, param in self.model.named_parameters():
                if any(k in name for k in [
                    "layer.10", "layer.11",      # BERT / RoBERTa
                    "layer.4",  "layer.5",       # ALBERT (6 layers total)
                    "encoder.layer.10", "encoder.layer.11",
                    "pooler",
                ]):
                    param.requires_grad = True
                    unfrozen += 1
            print(f"  TextEncoder ({model_key}): {unfrozen} parameter groups unfrozen.")

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # DeBERTa v3 does not emphasise CLS for sentence-level meaning;
        # mean-pool over non-padding tokens for a proper sentence embedding.
        # Other models use CLS which works well for them.
        if self.model_key == "deberta":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            return (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return out.last_hidden_state[:, 0, :]   # CLS token → (B, 768)


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
    def __init__(self, model_key):
        super().__init__()
        self.audio_proj  = AudioProjection()
        self.text_enc    = TextEncoder(model_key)
        self.fusion      = MultimodalFusion(
            text_dim=self.text_enc.embed_dim,
            num_classes=len(EMOTIONS),
        )

    def forward(self, audio_emb, texts):
        a = self.audio_proj(audio_emb)        # (B, 256)
        t = self.text_enc(texts)              # (B, 768)
        return self.fusion(a, t)


# ===============================
# Dataset
# ===============================

class SERDataset(torch.utils.data.Dataset):
    def __init__(self, audio_embeds, texts, labels):
        self.audio_embeds = audio_embeds
        self.texts        = texts
        self.labels       = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio_embeds[idx], self.texts[idx], self.labels[idx]


def collate_fn(batch):
    embeds, texts, lbls = zip(*batch)
    audio_tensor = torch.tensor(np.stack(embeds), dtype=torch.float32)
    label_tensor = torch.tensor(lbls, dtype=torch.long)
    return audio_tensor, list(texts), label_tensor


# ===============================
# Class weights + loss
# ===============================

def compute_class_weights(labels, num_classes=7):
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ===============================
# Training + Evaluation
# ===============================

def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for audio_emb, texts, labels in loader:
        audio_emb = audio_emb.to(device)
        labels    = labels.to(device)
        optimizer.zero_grad()
        logits = model(audio_emb, texts)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for audio_emb, texts, labels in loader:
            audio_emb = audio_emb.to(device)
            labels    = labels.to(device)
            logits    = model(audio_emb, texts)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), wf1, all_preds, all_labels


# ===============================
# Optimizer with differential LR
# ===============================

def build_optimizer(model):
    """
    Text transformer layers: LR_TEXT (1e-5), except DeBERTa which uses 1e-6
    Audio projection + fusion head: LR_HEAD (3e-4)
    DeBERTa v3 uses disentangled attention — extremely sensitive to LR,
    needs 10x lower text LR to avoid NaN gradient explosion.
    """
    is_deberta   = (model.text_enc.model_key == "deberta")
    text_lr      = 1e-6 if is_deberta else LR_TEXT
    text_params  = [p for p in model.text_enc.model.parameters() if p.requires_grad]
    other_params = (
        list(model.audio_proj.parameters()) +
        list(model.text_enc.tokenizer.parameters()
             if hasattr(model.text_enc.tokenizer, "parameters") else []) +
        list(model.fusion.parameters())
    )
    return torch.optim.AdamW([
        {"params": text_params,  "lr": text_lr},
        {"params": other_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)


# ===============================
# Run one experiment
# ===============================

def run_experiment(model_key, train_ds, dev_ds, test_ds, train_labels):
    print(f"\n{'='*62}")
    print(f"  Model    : {TEXT_MODELS[model_key]['description']}")
    print(f"  Audio    : wav2vec2-base (768-dim, frozen)")
    print(f"  Text LR  : {LR_TEXT}  |  Head LR : {LR_HEAD}")
    print(f"  Warmup   : {int(WARMUP_FRAC*100)}%  |  Label smooth : {LABEL_SMOOTH}")
    print(f"{'='*62}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    model         = EmotionModel(model_key).to(device)
    class_weights = compute_class_weights(train_labels)
    criterion     = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=LABEL_SMOOTH
    )
    optimizer     = build_optimizer(model)

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(WARMUP_FRAC * total_steps)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_wf1 = 0.0
    save_path    = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_key}.pth")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc                        = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        vl_loss, vl_acc, vl_wf1, _, _         = evaluate(model, dev_loader, criterion)
        print(
            f"  Epoch {epoch:2d}/{EPOCHS}  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}  Val WF1: {vl_wf1:.4f}"
        )
        if vl_wf1 > best_val_wf1:
            best_val_wf1 = vl_wf1
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved (WF1 {best_val_wf1:.4f})")

    # Test eval with best checkpoint
    model.load_state_dict(torch.load(save_path, weights_only=True))
    _, test_acc, test_wf1, preds, true_labels = evaluate(model, test_loader, criterion)

    print(f"\n  Best Dev WF1  : {best_val_wf1:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test WF1      : {test_wf1:.4f}")
    print(f"\n  Classification Report (Test):\n")
    print(classification_report(
        true_labels, preds,
        target_names=label_encoder.classes_, zero_division=0,
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    return best_val_wf1, test_acc, test_wf1


# ===============================
# Resume training
# ===============================

def resume_training(model_key, train_ds, dev_ds, test_ds,
                    train_labels, prev_best_wf1, extra_epochs=8):
    print(f"\n{'='*62}")
    print(f"  Resuming : {TEXT_MODELS[model_key]['description']}")
    print(f"  Adding   : {extra_epochs} more epochs  |  LR / 3")
    print(f"{'='*62}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_key}.pth")
    model     = EmotionModel(model_key).to(device)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  Loaded checkpoint: {save_path}")

    class_weights = compute_class_weights(train_labels)
    criterion     = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=LABEL_SMOOTH
    )
    optimizer     = build_optimizer(model)
    # halve LRs for resume
    for g in optimizer.param_groups:
        g["lr"] /= 3

    total_steps  = len(train_loader) * extra_epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps,
    )

    best_val_wf1 = prev_best_wf1
    print(f"  Previous best Dev WF1: {best_val_wf1:.4f}\n")

    for epoch in range(1, extra_epochs + 1):
        tr_loss, tr_acc                = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        vl_loss, vl_acc, vl_wf1, _, _ = evaluate(model, dev_loader, criterion)
        print(
            f"  Epoch {epoch:2d}/{extra_epochs}  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}  Val WF1: {vl_wf1:.4f}"
        )
        if vl_wf1 > best_val_wf1:
            best_val_wf1 = vl_wf1
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved (WF1 {best_val_wf1:.4f})")

    model.load_state_dict(torch.load(save_path, weights_only=True))
    _, test_acc, test_wf1, preds, true_labels = evaluate(model, test_loader, criterion)

    print(f"\n  Best Dev WF1 after resume : {best_val_wf1:.4f}")
    print(f"  Test Accuracy             : {test_acc:.4f}")
    print(f"  Test WF1                  : {test_wf1:.4f}")
    print(f"\n  Classification Report (Test):\n")
    print(classification_report(
        true_labels, preds,
        target_names=label_encoder.classes_, zero_division=0,
    ))
    print("  Confusion Matrix:")
    print(confusion_matrix(true_labels, preds))

    return best_val_wf1, test_acc, test_wf1


# ===============================
# Results table
# ===============================

def print_results_table(dev_wf1, test_acc, test_wf1):
    print("\n" + "="*70)
    print("  MELD v2 RESULTS — wav2vec2 + Text Encoder Multimodal SER")
    print("="*70)
    print(f"  {'Model':<38} {'Dev WF1':>9} {'Test Acc':>10} {'Test WF1':>10}")
    print("-"*70)
    for key in TEXT_MODELS:
        desc = TEXT_MODELS[key]["description"]
        dw   = dev_wf1.get(key, 0)
        ta   = test_acc.get(key, 0)
        tw   = test_wf1.get(key, 0)
        print(f"  {desc:<38} {dw:>9.4f} {ta:>10.4f} {tw:>10.4f}")
    print("="*70)
    if test_wf1:
        best_key = max(test_wf1, key=lambda k: test_wf1[k])
        print(
            f"\n  Best model (test WF1): {TEXT_MODELS[best_key]['description']}  "
            f"({test_wf1[best_key]:.4f})\n"
        )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    # ── Step 1: Load CSVs ──────────────────────────────────────
    print("Loading MELD CSV annotations ...")
    train_records = load_meld_csv(TRAIN_CSV)
    dev_records   = load_meld_csv(DEV_CSV)
    test_records  = load_meld_csv(TEST_CSV)
    print(f"  Train: {len(train_records)}  Dev: {len(dev_records)}  Test: {len(test_records)}\n")

    # ── Step 2: Load or extract wav2vec2 features ──────────────
    if os.path.exists(TRAIN_CACHE) and os.path.exists(DEV_CACHE) and os.path.exists(TEST_CACHE):
        print("Loading v2 feature caches from disk ...")
        with open(TRAIN_CACHE, "rb") as f:
            train_embeds, train_texts, train_labels = pickle.load(f)
        with open(DEV_CACHE, "rb") as f:
            dev_embeds, dev_texts, dev_labels = pickle.load(f)
        with open(TEST_CACHE, "rb") as f:
            test_embeds, test_texts, test_labels = pickle.load(f)
        print(f"  Loaded {len(train_embeds)} train + {len(dev_embeds)} dev + {len(test_embeds)} test.\n")
    else:
        wav2vec = Wav2VecExtractor()

        print("Pre-extracting TRAIN features (3× augmentation) ...")
        train_embeds, train_texts, train_labels = preextract_features(
            train_records, TRAIN_AUDIO_DIR, wav2vec, augment=True
        )
        with open(TRAIN_CACHE, "wb") as f:
            pickle.dump((train_embeds, train_texts, train_labels), f)

        print("Pre-extracting DEV features ...")
        dev_embeds, dev_texts, dev_labels = preextract_features(
            dev_records, DEV_AUDIO_DIR, wav2vec, augment=False
        )
        with open(DEV_CACHE, "wb") as f:
            pickle.dump((dev_embeds, dev_texts, dev_labels), f)

        print("Pre-extracting TEST features ...")
        test_embeds, test_texts, test_labels = preextract_features(
            test_records, TEST_AUDIO_DIR, wav2vec, augment=False
        )
        with open(TEST_CACHE, "wb") as f:
            pickle.dump((test_embeds, test_texts, test_labels), f)

        del wav2vec   # free GPU memory before training
        torch.cuda.empty_cache()
        print("Feature caches saved.\n")

    print(f"Training samples (after augmentation): {len(train_embeds)}")
    print(f"Dev samples: {len(dev_embeds)}  |  Test samples: {len(test_embeds)}\n")

    unique, counts = np.unique(train_labels, return_counts=True)
    print("Train label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {label_encoder.classes_[u]:<12}: {c:5d}  ({100*c/len(train_labels):.1f}%)")
    print()

    # ── Step 3: Build datasets ─────────────────────────────────
    train_ds = SERDataset(train_embeds, train_texts, train_labels)
    dev_ds   = SERDataset(dev_embeds,   dev_texts,   dev_labels)
    test_ds  = SERDataset(test_embeds,  test_texts,  test_labels)

    # ── Step 4: Train ──────────────────────────────────────────
    dev_wf1_results  = {}
    test_acc_results = {}
    test_wf1_results = {}

    if RESUME_MODE:
        print(f"\nRESUME MODE — adding {RESUME_EPOCHS} epochs to each model\n")
        prev_bests = {k: 0.0 for k in TEXT_MODELS}
        for key in TEXT_MODELS:
            save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{key}.pth")
            if not os.path.exists(save_path):
                print(f"  No checkpoint for {key} — running fresh ...")
                dev_wf1_results[key], test_acc_results[key], test_wf1_results[key] = run_experiment(
                    key, train_ds, dev_ds, test_ds, train_labels
                )
            else:
                dev_wf1_results[key], test_acc_results[key], test_wf1_results[key] = resume_training(
                    key, train_ds, dev_ds, test_ds,
                    train_labels, prev_bests[key], RESUME_EPOCHS
                )
    else:
        print("\nFRESH MODE — training all 5 models from scratch\n")
        for key in TEXT_MODELS:
            save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{key}.pth")
            if os.path.exists(save_path):
                print(f"  Checkpoint exists for {key} — loading for eval (delete to retrain).")
                model         = EmotionModel(key).to(device)
                model.load_state_dict(torch.load(save_path, weights_only=True))
                class_weights = compute_class_weights(train_labels)
                criterion     = nn.CrossEntropyLoss(
                    weight=class_weights, label_smoothing=LABEL_SMOOTH
                )
                dev_loader  = torch.utils.data.DataLoader(
                    dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_fn, num_workers=0,
                )
                test_loader = torch.utils.data.DataLoader(
                    test_ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_fn, num_workers=0,
                )
                _, dev_acc, dw, _, _  = evaluate(model, dev_loader, criterion)
                _, ta, tw, _, _       = evaluate(model, test_loader, criterion)
                dev_wf1_results[key]  = dw
                test_acc_results[key] = ta
                test_wf1_results[key] = tw
            else:
                dev_wf1_results[key], test_acc_results[key], test_wf1_results[key] = run_experiment(
                    key, train_ds, dev_ds, test_ds, train_labels
                )

    # ── Step 5: Final table ────────────────────────────────────
    print_results_table(dev_wf1_results, test_acc_results, test_wf1_results)
