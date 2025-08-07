#!/usr/bin/env python
"""
train_m4_unseen_model.py

Train on a subset of M4 JSONL files (held-in models/domains)
and test on the held-out models/domains, all without manual margin tuning.
Supports multi-way classification: “human” + multiple generator models.
"""
import os
import json
import glob
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import spacy
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from skdim.id import MLE, TwoNN

# ensure spaCy data and nltk punkt are available
nltk.download('punkt', quiet=True)
nltk.data.path.append("spacy_offline/nltk_data")

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
# 1) Data‐loading utilities
# ------------------------------------------------------------------------------

def find_jsonl_files(root):
    files = {}
    for fp in glob.glob(os.path.join(root, "*.jsonl")):
        files[Path(fp).name] = fp
    return files

def split_domain_model(fname):
    stem = Path(fname).stem
    domain, model = stem.rsplit("_", 1)
    return domain, model

def load_texts_and_labels(file_list, cls2idx):
    texts, labels = [], []
    for fp in file_list:
        fname = Path(fp).name
        domain, model = split_domain_model(fname)
        #count = 0

        for i, line in enumerate(open(fp, encoding="utf-8"), start=1):
            #if count >= 10:  # take only first 10 valid examples per file
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only process examples that have BOTH human_text and machine_text
            if "human_text" in rec and "machine_text" in rec:
                for text_raw, label in [
                    (rec["human_text"], "human"),
                    (rec["machine_text"], model)
                ]:
                    if not text_raw:
                        continue
                    # Normalize text: handle list vs string
                    if isinstance(text_raw, list):
                        text = " ".join(map(str, text_raw))
                    elif isinstance(text_raw, str):
                        text = text_raw
                    else:
                        continue  # unsupported type

                    text = text.strip()
                    if not text:
                        continue

                    # Append to data
                    texts.append(text)
                    labels.append(cls2idx.get(label, cls2idx["human"]))  # fallback if model not found
                    #count += 1

    return texts, labels

# ------------------------------------------------------------------------------
# 2) Feature & dataset classes
# ------------------------------------------------------------------------------

#def sentence_token_embeddings(text, embed_model):
 #   doc = nlp(text)
  #  tokens = [t.text for t in doc if t.is_alpha]
   # embs = embed_model.encode(tokens)
    #return embed_model.encode(tokens)

def sentence_token_embeddings(text, embed_model):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return embed_model.encode([text]) 
    return embed_model.encode(sentences)

def estimate_token_intrinsic_dim(text, embed_model, variance_threshold=0.95):
    embs = sentence_token_embeddings(text, embed_model)
    try:
        # Center the data
        embs_centered = embs - np.mean(embs, axis=0)
        pca = PCA()
        pca.fit(embs_centered)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        id_value = np.argmax(cumulative_variance >= variance_threshold) + 1

        #print(f"✅ Step 6: PCA-based Intrinsic Dim (≥ {variance_threshold*100:.0f}% variance): {id_value}")
        return float(id_value)
    except Exception as e:
        #print(f"❌ Step 7: PCA fitting failed: {e}")
        return 0.0

class SimCSEEmbedder:
    def __init__(self, model_dir, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model     = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
        self.model.eval()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            out = self.model(**toks).last_hidden_state.mean(dim=1)
        return out.cpu().numpy()

class ParagraphCombinedDataset(Dataset):
    def __init__(self, texts, labels, embedder):
        self.texts    = texts
        self.labels   = labels
        self.embedder = embedder

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]

        # linguistic features via spaCy + POS
        doc    = nlp(text)
        tokens = [t.text for t in doc if t.is_alpha]
        pos_   = pos_tag(tokens)
        words  = [w for w in tokens if len(w) > 1]

        feats = [
            np.mean([len(w) for w in words]) if words else 0.0,
            len(set(words)) / len(words) if words else 0.0,
            sum(1 for _,t in pos_ if t=='VBD'),
            sum(1 for _,t in pos_ if t=='VBP'),
            sum(1 for i in range(len(pos_)-1)
                if pos_[i][0].lower()=="have" and pos_[i+1][1]=='VBN'),
            sum(1 for w in tokens if w.lower() in {"i","we","me","us","our"}),
            sum(1 for w in tokens if w.lower() in {"you","your"}),
            sum(1 for w in tokens if w.lower() in {"he","she","they","him","her","them","their"}),
            sum(1 for w in tokens if w.lower() in {"can","could","may","might"}),
            sum(1 for w in tokens if w.lower() in {"must","should","ought"}),
            sum(1 for w in tokens if w.lower() in {"will","would","shall"}),
            sum(1 for s in doc.sents if "was" in s.text and "by" not in s.text),
            sum(1 for w in words if w.endswith(("tion","ment","ness","ity","ance","ence"))),
            sum(1 for i in range(len(tokens)-1)
                if tokens[i].lower()=="to" and pos_tag([tokens[i+1]])[0][1].startswith('V')),
            sum(1 for i in range(1,len(tokens)-1)
                if tokens[i].lower()=="and"
                and pos_tag([tokens[i-1]])[0][1][0]==pos_tag([tokens[i+1]])[0][1][0]),
            sum(1 for i in range(len(tokens))
                if tokens[i].lower() in {"and","but","or"} and
                i>0 and tokens[i-1].endswith(".")),
            sum(1 for w in tokens if w.lower() in {"well","now","anyway"}),
            sum(1 for w in tokens if "'" in w),
        ]
        ling = np.array(feats, dtype=np.float32)

        # intrinsic-dimension feature
        idim   = estimate_token_intrinsic_dim(text, self.embedder)
        id_feat = np.array([idim], dtype=np.float32)

        # SimCSE embedding
        emb = self.embedder.encode([text])[0]
        
        # concat & normalize
        feat = np.concatenate([ling, id_feat, emb], axis=0)
        feat = (feat - feat.mean())/(feat.std()+1e-6)

        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(id_feat, dtype=torch.float32),
        )

# ------------------------------------------------------------------------------
# 3) Model & loss
# ------------------------------------------------------------------------------

class MLPLearnedDistance(nn.Module):
    def __init__(self, in_dim, num_classes, proj_dim=128, hidden=64):
        super().__init__()
        self.projector    = nn.Linear(in_dim, proj_dim)
        self.mlp_distance = nn.Sequential(
        nn.Linear(proj_dim * 2, 128),
        nn.ReLU(),
        nn.LayerNorm(128),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

        # one prototype per class
        self.prototypes = nn.Parameter(torch.randn(num_classes, proj_dim))
        self.margin     = nn.Parameter(torch.tensor(0.2))

    def forward(self, x):
        proj = self.projector(x)
        if torch.isnan(proj).any():
            print("❌ NaN in projected features:", proj)
        return self.projector(x)

    def compute_distances(self, proj):
        B, D = proj.size()
        C    = self.prototypes.size(0)
        xp   = proj.unsqueeze(1).expand(B,C,D)
        pp   = self.prototypes.unsqueeze(0).expand(B,C,D)
        pair = torch.cat([xp,pp], dim=2)
        out = self.mlp_distance(pair).squeeze(-1)

        if torch.isnan(out).any():
            print("❌ NaN in distance output:", out)
            print("➡️ Pair input to distance MLP:", pair)
        return self.mlp_distance(pair).squeeze(-1)

def iso_prototype_loss(feats, labels, model, weights):
    proj   = model(feats)                            # (B, proj_dim)
    logits = -model.compute_distances(proj)          # (B, C)
    # subtract learned margin on true-class logit
    logits[torch.arange(len(labels)), labels] -= model.margin.clamp(0.0,1.0)
    ce     = F.cross_entropy(logits, labels, reduction='none')

    return (weights * ce).mean()

# ------------------------------------------------------------------------------
# 4) Training + evaluation loops
# ------------------------------------------------------------------------------

def train_and_validate(train_files, embedder, in_dim, num_classes,
                       epochs=10, batch_size=8, lr=2e-5):
    # build train/val split
    texts, labels = load_texts_and_labels(train_files, cls2idx)
    from sklearn.model_selection import train_test_split
    t_texts, v_texts, t_labels, v_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_ds = ParagraphCombinedDataset(t_texts, t_labels, embedder)
    val_ds   = ParagraphCombinedDataset(v_texts, v_labels, embedder)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = MLPLearnedDistance(in_dim, num_classes).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs+1):
        model.train()
        tot_loss, corr = 0.0, 0
        for feats, lbls, w in train_ld:
            feats, lbls, w = feats.to(device), lbls.to(device), w.to(device)
            loss = iso_prototype_loss(feats, lbls, model, w.clamp(0,1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
            with torch.no_grad():
                logits = model.compute_distances(model(feats))
                corr  += (logits.argmin(1)==lbls).sum().item()
        tr_acc = corr/len(train_ld.dataset)
        print(f"[Ep{ep}] Train Loss={tot_loss:.4f} Acc={tr_acc:.3f}")

        model.eval()
        corr = 0
        with torch.no_grad():
            for feats, lbls, w in val_ld:
                feats, lbls = feats.to(device), lbls.to(device)
                logits = model.compute_distances(model(feats))
                corr  += (logits.argmin(1)==lbls).sum().item()
        va_acc = corr/len(val_ld.dataset)
        print(f"        Val Acc={va_acc:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/final_m4_model.pt")
    print("✅ Model saved to checkpoints/final_m4_model.pt")
    return model

def evaluate_on_files(test_files, embedder, model, batch_size=8):
    texts, labels = load_texts_and_labels(test_files, cls2idx)
    ds  = ParagraphCombinedDataset(texts, labels, embedder)
    ld  = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    corr = 0
    with torch.no_grad():
        for feats, lbls, w in ld:
            feats,lbls = feats.to(device), lbls.to(device)
            logits = model.compute_distances(model(feats))
            corr  += (logits.argmin(1)==lbls).sum().item()
    acc = corr/len(ds)
    print(f"🎯 Test  Acc={acc:.3f}")

# ------------------------------------------------------------------------------
# 5) CLI entrypoint
# ------------------------------------------------------------------------------

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Train on held-in M4 models/domains, test on held-out ones"
    )
    parser.add_argument("--data-root",     default="M4/data",
                        help="folder containing M4 .jsonl files")
    parser.add_argument("--train-models", nargs="+", required=True,
                        help="model names to TRAIN on (e.g. davinci chatGPT cohere)")
    parser.add_argument("--test-models",  nargs="+", required=True,
                        help="model names to TEST on")
    parser.add_argument("--train-domains", nargs="+", required=True,
                        help="domain prefixes to TRAIN on (e.g. wikipedia reddit)")
    parser.add_argument("--test-domains",  nargs="+", required=True,
                        help="domain prefixes to TEST on")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # load spaCy once
    print("🧠 Loading spaCy English model...")
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy loaded.")

    # discover all files
    all_files = find_jsonl_files(args.data_root)

    # build class list: human + union of train/test models
    classes = ["human"] + sorted(set(args.train_models + args.test_models))
    cls2idx = { c:i for i,c in enumerate(classes) }
    num_classes = len(classes)

    # partition train/test file lists
    train_fps, test_fps = [], []
    for name, fp in all_files.items():
        dom, mdl = split_domain_model(name)
        if dom in args.train_domains and mdl in args.train_models:
            train_fps.append(fp)
        if dom in args.test_domains  and mdl in args.test_models:
            test_fps.append(fp)

    print(f"📂 Will TRAIN on {len(train_fps)} files, TEST on {len(test_fps)} files.")
    print(f"🔖 Classes: {classes}")

    # load embedder
    print("🔗 Loading SimCSE embedder…")
    embed_model = SimCSEEmbedder(model_dir="SimCSE-RoBERTa", device=device)
    print("✅ Embedder ready.")

    # compute input dimension
    in_dim = 18 + 1 + embed_model.encode(["hi"]).shape[1]

    # train + validate
    model = train_and_validate(
        train_fps,
        embed_model,
        in_dim,
        num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # final held-out evaluation
    evaluate_on_files(test_fps, embed_model, model, batch_size=args.batch_size)
