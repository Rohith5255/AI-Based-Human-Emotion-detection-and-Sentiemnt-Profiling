# -*- coding: utf-8 -*-
"""
CIS Project — Multimodal SER on MELD  (Version 3c)
====================================================
v3c = v2 + CrossAttentionFusion (only change from v2):

  CrossAttentionFusion replaces v2's sigmoid gate  (only change)
  ─────────────────────────────────────────────────────────────
  v2:  concat(audio,text) → sigmoid gate element-wise → MLP → 7 classes
  v3c: audio queries text  (audio-to-text MHA, d_model=256, 4 heads)
       text queries audio  (text-to-audio MHA, d_model=256, 4 heads)
       residual + LayerNorm on each output
       concat(256+256=512) → MLP → 7 classes

Everything else is IDENTICAL to v2:
  • wav2vec2 fully frozen, pre-extracted embeddings
  • Reuses D:\\SEAD\\meld_cache\\meld_v2_*.pkl  (no re-extraction)
  • CrossEntropyLoss + class weights + label_smoothing=0.1
  • Batch size 32, LR_TEXT=1e-5, LR_HEAD=3e-4
  • Linear warmup 10% + cosine annealing
  • 15 epochs, WEIGHT_DECAY=1e-4
  • Same 4 text encoders (DeBERTa excluded)

Saves to: D:\\SEAD\\meld_cache\\saved_models_meld_v3c\\

v2 baseline (RoBERTa): Test WF1 = 0.5009, Test Acc = 50.5%
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

warnings.filterwarnings("ignore")

# ── ffmpeg path ───────────────────────────────────────────────────────────────
_ffmpeg_bin = (
    r"C:\Users\rohit\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
)
if _ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

# ================================================================
# PATHS
# ================================================================
MELD_ROOT       = r"C:\Users\rohit\MELD.Raw\MELD.Raw"
TRAIN_AUDIO_DIR = os.path.join(MELD_ROOT, "train_splits")
DEV_AUDIO_DIR   = os.path.join(MELD_ROOT, "dev_splits_complete")
TEST_AUDIO_DIR  = os.path.join(MELD_ROOT, "output_repeated_splits_test")
TRAIN_CSV       = os.path.join(MELD_ROOT, "train_sent_emo.csv")
DEV_CSV         = os.path.join(MELD_ROOT, "dev_sent_emo.csv")
TEST_CSV        = os.path.join(MELD_ROOT, "test_sent_emo.csv")

TRAIN_CACHE    = r"D:\SEAD\meld_cache\meld_v2_train_features.pkl"
DEV_CACHE      = r"D:\SEAD\meld_cache\meld_v2_dev_features.pkl"
TEST_CACHE     = r"D:\SEAD\meld_cache\meld_v2_test_features.pkl"
MODEL_SAVE_DIR = r"D:\SEAD\meld_cache\saved_models_meld_v3c"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================================================================
# Hyperparameters  — identical to v2
# ================================================================
TARGET_SR    = 16000
EPOCHS       = 15
BATCH_SIZE   = 32
LR_TEXT      = 1e-5
LR_HEAD      = 3e-4
WARMUP_FRAC  = 0.10
LABEL_SMOOTH = 0.10
WEIGHT_DECAY = 1e-4

WAV2VEC_MODEL = "facebook/wav2vec2-base"
WAV2VEC_DIM   = 768

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
label_encoder = LabelEncoder()
label_encoder.fit(EMOTIONS)

MELD_EMOTION_MAP = {e: e for e in EMOTIONS}

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
    # DeBERTa v3 excluded — NaN instability documented in v2.
    "albert": {
        "name":        "albert-base-v2",
        "embed_dim":   768,
        "description": "ALBERT base v2",
    },
}


# ================================================================
# Audio utilities  (used only if caches are missing)
# ================================================================

def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    min_len   = TARGET_SR // 2
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))
    return audio


def augment_audio_3x(audio):
    augmented = []
    for rate in (0.9, 1.1):
        try:
            augmented.append(librosa.effects.time_stretch(audio, rate=rate))
        except Exception:
            augmented.append(audio.copy())
    return augmented


def build_audio_path(audio_dir, dialogue_id, utterance_id):
    base = f"dia{dialogue_id}_utt{utterance_id}"
    for ext in [".wav", ".mp4"]:
        path = os.path.join(audio_dir, base + ext)
        if os.path.exists(path):
            return path
    return None


# ================================================================
# CSV loading
# ================================================================

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


# ================================================================
# wav2vec2 Feature Extractor  (frozen — identical to v2)
# ================================================================

class Wav2VecExtractor:
    def __init__(self):
        print(f"  Loading {WAV2VEC_MODEL} for audio feature extraction ...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC_MODEL)
        self.model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("  wav2vec2-base loaded (frozen, inference only).\n")

    @torch.no_grad()
    def embed(self, audio_np):
        inputs = self.feature_extractor(
            audio_np, sampling_rate=TARGET_SR,
            return_tensors="pt", padding=True,
        )
        out = self.model(inputs.input_values.to(device))
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


# ================================================================
# Feature pre-extraction  (only if caches missing)
# ================================================================

def preextract_features(records, audio_dir, wav2vec, augment=False):
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
                for aug in augment_audio_3x(audio):
                    try:
                        audio_embeds.append(wav2vec.embed(aug))
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


# ================================================================
# Model Architecture
# ================================================================

class AudioProjection(nn.Module):
    """768 → 256 MLP  (identical to v2)."""
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
        return self.net(x)   # (B, 256)


class TextEncoder(nn.Module):
    """Top-2 layers + pooler unfrozen; CLS output  (identical to v2)."""
    def __init__(self, model_key):
        super().__init__()
        cfg            = TEXT_MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
        self.model     = AutoModel.from_pretrained(cfg["name"], use_safetensors=True)
        self.embed_dim = cfg["embed_dim"]
        self.model_key = model_key

        for param in self.model.parameters():
            param.requires_grad = False

        unfrozen = 0
        for name, param in self.model.named_parameters():
            if any(k in name for k in [
                "layer.10", "layer.11",
                "layer.4",  "layer.5",
                "encoder.layer.10", "encoder.layer.11",
                "pooler",
            ]):
                param.requires_grad = True
                unfrozen += 1
        print(f"  TextEncoder ({model_key}): {unfrozen} parameter tensors unfrozen.")

    def forward(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out    = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]   # CLS → (B, 768)


class CrossAttentionFusion(nn.Module):
    """
    Replaces v2's sigmoid gate with bidirectional cross-attention.

    audio (B,256) and text (B,768) projected to d_model=256.
    audio-to-text : audio queries attend to text keys/values
    text-to-audio : text queries attend to audio keys/values
    Residual + LayerNorm on each → concat(256+256=512) → MLP → num_classes.
    """
    def __init__(self, audio_dim=256, text_dim=768,
                 d_model=256, num_heads=4, dropout=0.1, num_classes=7):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj  = nn.Linear(text_dim,  d_model)

        self.attn_a2t = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_t2a = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, audio_vec, text_vec):
        a = self.audio_proj(audio_vec).unsqueeze(1)   # (B, 1, 256)
        t = self.text_proj(text_vec).unsqueeze(1)     # (B, 1, 256)

        a_ctx, _ = self.attn_a2t(a, t, t)
        t_ctx, _ = self.attn_t2a(t, a, a)

        a_out = self.norm_a(a + a_ctx).squeeze(1)    # (B, 256)
        t_out = self.norm_t(t + t_ctx).squeeze(1)    # (B, 256)

        return self.classifier(torch.cat([a_out, t_out], dim=1))


class EmotionModel(nn.Module):
    def __init__(self, model_key):
        super().__init__()
        self.audio_proj = AudioProjection()
        self.text_enc   = TextEncoder(model_key)
        self.fusion     = CrossAttentionFusion(
            text_dim=self.text_enc.embed_dim,
            num_classes=len(EMOTIONS),
        )

    def forward(self, audio_emb, texts):
        a = self.audio_proj(audio_emb)   # (B, 256)
        t = self.text_enc(texts)         # (B, 768)
        return self.fusion(a, t)


# ================================================================
# Dataset  (identical to v2)
# ================================================================

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


# ================================================================
# Class weights  (identical to v2)
# ================================================================

def compute_class_weights(labels, num_classes=7):
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ================================================================
# Optimizer  (identical to v2)
# ================================================================

def build_optimizer(model):
    text_params  = [p for p in model.text_enc.model.parameters() if p.requires_grad]
    other_params = list(model.audio_proj.parameters()) + list(model.fusion.parameters())
    return torch.optim.AdamW([
        {"params": text_params,  "lr": LR_TEXT},
        {"params": other_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)


# ================================================================
# Training & Evaluation  (identical to v2)
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for audio_emb, texts, labels in loader:
        audio_emb = audio_emb.to(device)
        labels    = labels.to(device)
        optimizer.zero_grad()
        logits    = model(audio_emb, texts)
        loss      = criterion(logits, labels)
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


# ================================================================
# Run one experiment
# ================================================================

def run_experiment(model_key, train_ds, dev_ds, test_ds, train_labels):
    print(f"\n{'='*65}")
    print(f"  Model    : {TEXT_MODELS[model_key]['description']}")
    print(f"  Audio    : wav2vec2-base (768-dim, FROZEN, pre-extracted)")
    print(f"  Fusion   : CrossAttentionFusion (d_model=256, 4 heads)")
    print(f"  Loss     : CrossEntropyLoss + class weights + label_smooth={LABEL_SMOOTH}")
    print(f"  Text LR  : {LR_TEXT}  |  Head LR : {LR_HEAD}")
    print(f"  Warmup   : {int(WARMUP_FRAC*100)}%  |  Epochs : {EPOCHS}")
    print(f"{'='*65}")

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
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    optimizer     = build_optimizer(model)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_wf1 = 0.0
    save_path    = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_key}.pth")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc                = train_one_epoch(model, train_loader, criterion, optimizer, scheduler)
        vl_loss, vl_acc, vl_wf1, _, _ = evaluate(model, dev_loader, criterion)
        print(
            f"  Epoch {epoch:2d}/{EPOCHS}  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}  Val WF1: {vl_wf1:.4f}"
        )
        if vl_wf1 > best_val_wf1:
            best_val_wf1 = vl_wf1
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best saved (WF1 {best_val_wf1:.4f})")

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


# ================================================================
# Results table
# ================================================================

def print_results_table(dev_wf1, test_acc, test_wf1):
    print("\n" + "=" * 70)
    print("  MELD v3c RESULTS — wav2vec2 frozen + CrossAttn + CE+LabelSmooth")
    print("=" * 70)
    print(f"  {'Model':<38} {'Dev WF1':>9} {'Test Acc':>10} {'Test WF1':>10}")
    print("-" * 70)
    for key in TEXT_MODELS:
        desc = TEXT_MODELS[key]["description"]
        dw   = dev_wf1.get(key,  0)
        ta   = test_acc.get(key, 0)
        tw   = test_wf1.get(key, 0)
        print(f"  {desc:<38} {dw:>9.4f} {ta:>10.4f} {tw:>10.4f}")
    print("=" * 70)
    if test_wf1:
        best_key = max(test_wf1, key=lambda k: test_wf1[k])
        print(
            f"\n  Best model (test WF1): {TEXT_MODELS[best_key]['description']}  "
            f"({test_wf1[best_key]:.4f})"
        )
    print("  v2 baseline (RoBERTa): Test WF1 = 0.5009\n")


# ================================================================
# Smoke test
# ================================================================

def run_smoke_test():
    """5 samples from v2 test cache, BERT, one forward pass, confirm no NaN."""
    print("=" * 55)
    print("  SMOKE TEST — 5 samples, BERT, 1 forward pass")
    print("=" * 55)

    if not os.path.exists(TEST_CACHE):
        print(f"\n[FAIL] Test cache not found: {TEST_CACHE}")
        return

    print(f"\nLoading 5 samples from v2 test cache ...")
    with open(TEST_CACHE, "rb") as f:
        test_embeds, test_texts, test_labels = pickle.load(f)

    embeds   = test_embeds[:5]
    texts    = test_texts[:5]
    labels   = test_labels[:5]

    for i, (t, l) in enumerate(zip(texts, labels)):
        print(f"  [{i}] label={label_encoder.classes_[l]:<8}  text='{t}'")

    labels_np = np.array(labels)
    audio_t   = torch.tensor(np.stack(embeds), dtype=torch.float32).to(device)
    labels_t  = torch.tensor(labels_np, dtype=torch.long).to(device)

    print(f"\nBuilding EmotionModel(bert) ...")
    model = EmotionModel("bert").to(device)
    model.train()

    class_weights = compute_class_weights(labels_np)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

    print("Running forward pass ...")
    logits = model(audio_t, list(texts))
    loss   = criterion(logits, labels_t)

    loss_val = loss.item()
    is_nan   = loss_val != loss_val

    print(f"\n  Logits shape : {tuple(logits.shape)}")
    print(f"  Loss value   : {loss_val:.6f}")
    print(f"  NaN?         : {is_nan}")
    print(f"  Predicted    : {torch.argmax(logits, dim=1).cpu().tolist()}")
    print(f"  True labels  : {labels_np.tolist()}")

    if is_nan:
        print("\n[FAIL] Loss is NaN.\n")
    else:
        print("\n[PASS] Forward pass clean — loss is finite and not NaN.\n")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run 5-sample smoke test and exit.")
    args = parser.parse_args()

    if args.smoke:
        run_smoke_test()
        raise SystemExit(0)

    # Step 1 — Load CSVs
    print("Loading MELD CSV annotations ...")
    train_records = load_meld_csv(TRAIN_CSV)
    dev_records   = load_meld_csv(DEV_CSV)
    test_records  = load_meld_csv(TEST_CSV)
    print(f"  Train: {len(train_records)}  Dev: {len(dev_records)}  Test: {len(test_records)}\n")

    # Step 2 — Load v2 embedding caches
    if os.path.exists(TRAIN_CACHE) and os.path.exists(DEV_CACHE) and os.path.exists(TEST_CACHE):
        print("Loading v2 embedding caches ...")
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
        del wav2vec
        torch.cuda.empty_cache()
        print("Caches saved.\n")

    print(f"Training samples (after augmentation): {len(train_embeds)}")
    print(f"Dev samples: {len(dev_embeds)}  |  Test samples: {len(test_embeds)}\n")

    unique, counts = np.unique(train_labels, return_counts=True)
    print("Train label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {label_encoder.classes_[u]:<12}: {c:5d}  ({100*c/len(train_labels):.1f}%)")
    print()

    # Step 3 — Build datasets
    train_ds = SERDataset(train_embeds, train_texts, train_labels)
    dev_ds   = SERDataset(dev_embeds,   dev_texts,   dev_labels)
    test_ds  = SERDataset(test_embeds,  test_texts,  test_labels)

    # Step 4 — Train all models
    dev_wf1_results  = {}
    test_acc_results = {}
    test_wf1_results = {}

    print("\nFRESH MODE — training all models from scratch\n")
    for key in TEXT_MODELS:
        save_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{key}.pth")
        if os.path.exists(save_path):
            print(f"  Checkpoint exists for {key} — loading for eval (delete to retrain).")
            model         = EmotionModel(key).to(device)
            model.load_state_dict(torch.load(save_path, weights_only=True))
            class_weights = compute_class_weights(train_labels)
            criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
            dev_loader    = torch.utils.data.DataLoader(
                dev_ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )
            test_loader   = torch.utils.data.DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )
            _, _, dw, _, _  = evaluate(model, dev_loader,  criterion)
            _, ta, tw, _, _ = evaluate(model, test_loader, criterion)
            dev_wf1_results[key]  = dw
            test_acc_results[key] = ta
            test_wf1_results[key] = tw
        else:
            dev_wf1_results[key], test_acc_results[key], test_wf1_results[key] = run_experiment(
                key, train_ds, dev_ds, test_ds, train_labels
            )

    # Step 5 — Final table
    print_results_table(dev_wf1_results, test_acc_results, test_wf1_results)
