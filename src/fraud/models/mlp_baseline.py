"""
Tabular MLP baseline (October Task #14), the team's first deep learning model.

Architecture choices follow docs/dl_literature_review.md:

- Inputs: the same fe_v2 features as the XGBoost baseline.
- Categorical columns (text fields plus the numeric codes card1-5 and
  addr1-2) get learned embeddings. Vocabularies come from the train split
  only; values seen fewer than --min-count times share one "rare" slot, and
  missing has its own slot.
- Numeric columns are quantile-normalized (fitted on train) and missing values
  become 0, with a 0/1 missing flag per distinct missing pattern (columns that
  are always missing together share one flag, which collapses the V blocks).
- Body: Linear -> ReLU -> Dropout blocks.
- Imbalance: weighted BCE (pos_weight = legit/fraud ratio, or its square root)
  or focal loss.
- Early stopping on validation AUPRC; the best epoch's weights are kept.

The blind test split is never loaded.

    python -m fraud.models.mlp_baseline
    python -m fraud.models.mlp_baseline --loss focal --epochs 20
"""

from __future__ import annotations

import argparse
import copy
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from fraud.models import xgboost_baseline as xb

CODE_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
DROP = [xb.TARGET, xb.ID_COL, xb.TIME_COL]


# ---------------------------------------------------------------------------
# Preprocessing (fit on train only)
# ---------------------------------------------------------------------------

class TabularPreprocessor:
    def __init__(self, min_count: int = 20, max_emb_dim: int = 32, qt_sample: int = 200_000, seed: int = 42):
        self.min_count, self.max_emb_dim, self.qt_sample, self.seed = min_count, max_emb_dim, qt_sample, seed

    @staticmethod
    def _as_key(s: pd.Series) -> pd.Series:
        if pd.api.types.is_float_dtype(s):
            s = s.astype("Int64")
        return s.astype("string")

    def fit(self, df: pd.DataFrame):
        cols = [c for c in df.columns if c not in DROP]
        self.cat_cols = [c for c in cols if c in CODE_COLS or not pd.api.types.is_numeric_dtype(df[c])]
        self.num_cols = [c for c in cols if c not in self.cat_cols]

        self.vocab = {}
        for c in self.cat_cols:
            vc = self._as_key(df[c]).value_counts(dropna=True)
            keep = vc[vc >= self.min_count].index
            # 0 = missing, 1 = rare or unseen, 2.. = frequent values
            self.vocab[c] = pd.Series(np.arange(2, len(keep) + 2), index=keep)
        self.emb_dims = [(len(self.vocab[c]) + 2, min(self.max_emb_dim, int(round(1.6 * (len(self.vocab[c]) + 2) ** 0.56))))
                         for c in self.cat_cols]

        miss = df[self.num_cols].isna()
        has_missing = miss.any()
        patterns = {}
        for c in has_missing[has_missing].index:
            key = hash(miss[c].values.tobytes())
            patterns.setdefault(key, c)
        self.flag_cols = list(patterns.values())  # one representative column per missing pattern

        sample = df[self.num_cols]
        if len(sample) > self.qt_sample:
            sample = sample.sample(n=self.qt_sample, random_state=self.seed)
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=10**9,
                                      random_state=self.seed)
        self.qt.fit(sample.astype("float32").values)
        return self

    def transform(self, df: pd.DataFrame):
        cat = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            key = self._as_key(df[c])
            idx = key.map(self.vocab[c])
            cat[:, j] = np.where(key.isna(), 0, idx.fillna(1)).astype(np.int64)
        num = self.qt.transform(df[self.num_cols].astype("float32").values).astype(np.float32)
        np.nan_to_num(num, copy=False, nan=0.0)
        flags = df[self.flag_cols].isna().values.astype(np.float32)
        return cat, np.hstack([num, flags])

    @property
    def n_dense(self):
        return len(self.num_cols) + len(self.flag_cols)


# ---------------------------------------------------------------------------
# Model and losses
# ---------------------------------------------------------------------------

class TabularMLP(nn.Module):
    def __init__(self, emb_dims, n_dense, hidden=(512, 256, 128), dropout=0.25, emb_dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d) for n, d in emb_dims])
        self.emb_drop = nn.Dropout(emb_dropout)
        width = sum(d for _, d in emb_dims) + n_dense
        layers = []
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, cat, dense):
        e = [emb(cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([self.emb_drop(torch.cat(e, dim=1)), dense], dim=1) if e else dense
        return self.body(x).squeeze(1)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017): down-weights easy examples."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(y == 1, p, 1 - p)
        a = torch.where(y == 1, torch.full_like(y, self.alpha), torch.full_like(y, 1 - self.alpha))
        return (a * (1 - pt) ** self.gamma * bce).mean()


def make_loss(kind: str, y_train: np.ndarray):
    ratio = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
    if kind == "focal":
        return FocalLoss(), {"loss": "focal", "gamma": 2.0, "alpha": 0.75}
    w = {"bce_weighted": ratio, "bce_sqrt": ratio ** 0.5, "bce": 1.0}[kind]
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w)), {"loss": kind, "pos_weight": round(w, 3)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def predict(model, cat, dense, batch=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(cat), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(cat[i:i + batch]), torch.from_numpy(dense[i:i + batch]))))
    return torch.cat(out).numpy()


def train_mlp(train: pd.DataFrame, valid: pd.DataFrame, *, loss="bce_weighted", epochs=15, batch_size=1024,
              lr=1e-3, weight_decay=1e-5, hidden=(512, 256, 128), dropout=0.25, patience=3, min_count=20,
              seed=42, threads=None, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if threads:
        torch.set_num_threads(threads)
    pre = TabularPreprocessor(min_count=min_count, seed=seed).fit(train)
    cat_tr, den_tr = pre.transform(train)
    cat_va, den_va = pre.transform(valid)
    y_tr = train[xb.TARGET].values.astype(np.float32)
    y_va = valid[xb.TARGET].values.astype(int)

    model = TabularMLP(pre.emb_dims, pre.n_dense, hidden, dropout)
    criterion, loss_info = make_loss(loss, y_tr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=int(np.ceil(len(y_tr) / batch_size)))
    cat_t, den_t, y_t = torch.from_numpy(cat_tr), torch.from_numpy(den_tr), torch.from_numpy(y_tr)

    best, best_state, best_epoch, bad, history = -1.0, None, -1, 0, []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y_t))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            l = criterion(model(cat_t[idx], den_t[idx]), y_t[idx])
            l.backward()
            opt.step()
            sched.step()
            total += float(l) * len(idx)
        score = predict(model, cat_va, den_va)
        ap = float(average_precision_score(y_va, score))
        history.append({"epoch": epoch + 1, "train_loss": total / len(y_t), "valid_auprc": ap,
                        "seconds": round(time.time() - start, 1)})
        if verbose:
            print(f"epoch {epoch + 1:2d} loss {total / len(y_t):.4f} valid AUPRC {ap:.4f} ({time.time() - start:.0f}s)",
                  flush=True)
        if ap > best:
            best, best_state, best_epoch, bad = ap, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    valid_score = predict(model, cat_va, den_va)
    info = {
        "architecture": {"hidden": list(hidden), "dropout": dropout, "embeddings": len(pre.emb_dims),
                         "embedding_width": sum(d for _, d in pre.emb_dims), "dense_inputs": pre.n_dense,
                         "missing_flags": len(pre.flag_cols),
                         "parameters": int(sum(p.numel() for p in model.parameters()))},
        "training": {"optimizer": "AdamW", "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
                     "schedule": "OneCycle", "epochs_max": epochs, "best_epoch": best_epoch, "patience": patience,
                     "min_count": min_count, **loss_info},
        "history": history,
        "train_seconds": round(time.time() - start, 1),
    }
    return model, pre, valid_score, info


def run(data_path=None, split_dir=None, registry=None, report_dir=None, notes="", **kw):
    df = xb.load_merged(data_path)
    split = xb.assign_split(df, split_dir)
    train, valid = df[split == "train"], df[split == "valid"]
    del df
    gc.collect()
    train, [valid] = xb.add_features_v2(train, [valid])
    model, pre, score, info = train_mlp(train, valid, **kw)
    metrics = xb.evaluate(valid[xb.TARGET].values, score, amount=valid[xb.AMOUNT_COL].values)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "mlp",
        "task": "Oct #14 DL baseline",
        "feature_set": "fe_v2",
        "n_features": len(pre.cat_cols) + len(pre.num_cols),
        "n_categorical": len(pre.cat_cols),
        "params": {**info["architecture"], **info["training"]},
        "split_id": split.attrs["source"],
        "split_sizes": {k: int((split == k).sum()) for k in xb.SPLIT_DAYS},
        "seed": kw.get("seed", 42),
        "git_commit": xb._git_commit(),
        "train_seconds": info["train_seconds"],
        "history": info["history"],
        "metrics": {"valid": metrics},
        "notes": notes,
    }
    path = xb.log_run(record, registry)
    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        tag = info["training"]["loss"]
        np.save(report_dir / f"mlp_{tag}_valid_scores.npy", score)
        torch.save(model.state_dict(), report_dir / f"mlp_{tag}.pt")
    print(f"AUPRC {metrics['auprc']:.4f} ROC-AUC {metrics['roc_auc']:.4f} "
          f"R@1% {metrics['recall_at_1pct_review']:.3f} R@5% {metrics['recall_at_5pct_review']:.3f} -> {path}")
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description="Tabular MLP baseline (Oct Task #14)")
    p.add_argument("--data", type=Path)
    p.add_argument("--split-dir", type=Path)
    p.add_argument("--registry", type=Path)
    p.add_argument("--report-dir", type=Path)
    p.add_argument("--loss", choices=["bce_weighted", "bce_sqrt", "bce", "focal"], default="bce_weighted")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", default="512,256,128")
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int)
    p.add_argument("--notes", default="")
    a = p.parse_args(argv)
    run(a.data, a.split_dir, a.registry, a.report_dir, a.notes, loss=a.loss, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr, hidden=tuple(int(h) for h in a.hidden.split(",")), dropout=a.dropout,
        patience=a.patience, min_count=a.min_count, seed=a.seed, threads=a.threads)


if __name__ == "__main__":
    main()
