"""
FT-Transformer baseline for October Task #15.

This model treats each tabular feature as a token and uses self-attention
to learn interactions between transaction features.

The preprocessing and chronological split reuse the same pipeline as the
MLP baseline so the two deep-learning models are directly comparable.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from fraud.models import xgboost_baseline as xb
from fraud.models.mlp_baseline import FocalLoss, TabularPreprocessor


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_ft_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    max_numeric: int = 100,
    max_missing_rate: float = 0.80,
    corr_threshold: float = 0.98,
):
    """
    Keep all categorical features, then select a compact set of numerical
    features using training data only.

    Numerical features are filtered by:
    1. missingness,
    2. non-zero variance,
    3. very high pairwise correlation,
    4. a maximum feature count.

    No validation labels or blind-test data are used for selection.
    """

    drop = {
        xb.TARGET,
        xb.ID_COL,
        xb.TIME_COL,
    }

    feature_cols = [
        c for c in train.columns
        if c not in drop
    ]

    # Match the categorical-code treatment used by the MLP baseline.
    categorical_code_cols = {
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
    }

    categorical_cols = [
        c
        for c in feature_cols
        if (
            c in categorical_code_cols
            or not pd.api.types.is_numeric_dtype(train[c])
        )
    ]

    numeric_cols = [
        c
        for c in feature_cols
        if c not in categorical_cols
    ]

    # Remove numerical columns that are mostly missing.
    missing_rate = train[numeric_cols].isna().mean()

    numeric_cols = [
        c
        for c in numeric_cols
        if missing_rate[c] <= max_missing_rate
    ]

    # Remove constant or effectively constant numerical columns.
    variances = train[numeric_cols].var(skipna=True)

    numeric_cols = [
        c
        for c in numeric_cols
        if (
            pd.notna(variances[c])
            and variances[c] > 0
        )
    ]

    # Rank deterministically before correlation filtering.
    numeric_cols = sorted(
        numeric_cols,
        key=lambda c: float(variances[c]),
        reverse=True,
    )

    selected_numeric = []

    # Remove numerical features that are almost duplicates of one already kept.
    for col in numeric_cols:
        if len(selected_numeric) >= max_numeric:
            break

        if not selected_numeric:
            selected_numeric.append(col)
            continue

        correlations = (
            train[selected_numeric]
            .corrwith(train[col])
            .abs()
        )

        max_corr = correlations.max(skipna=True)

        if pd.isna(max_corr) or max_corr < corr_threshold:
            selected_numeric.append(col)

    selected = categorical_cols + selected_numeric

    selection_info = {
        "categorical_features": len(categorical_cols),
        "numeric_features": len(selected_numeric),
        "total_features": len(selected),
        "selected_columns": selected,
    }

    return (
        train[selected + [xb.TARGET]].copy(),
        valid[selected + [xb.TARGET]].copy(),
        selection_info,
    )


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------

def prepare_ft_data(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    max_numeric: int = 100,
):
    """
    Select FT-Transformer features and apply the same preprocessing class
    used by the MLP baseline.
    """

    train_small, valid_small, selection_info = select_ft_features(
        train,
        valid,
        max_numeric=max_numeric,
    )

    preprocessor = TabularPreprocessor().fit(train_small)

    cat_train, dense_train = preprocessor.transform(
        train_small
    )

    cat_valid, dense_valid = preprocessor.transform(
        valid_small
    )

    y_train = train_small[xb.TARGET].to_numpy(
        dtype=np.float32
    )

    y_valid = valid_small[xb.TARGET].to_numpy(
        dtype=np.int64
    )

    categorical_cardinalities = [
        n_categories
        for n_categories, _ in preprocessor.emb_dims
    ]

    return {
        "cat_train": cat_train,
        "num_train": dense_train,
        "y_train": y_train,
        "cat_valid": cat_valid,
        "num_valid": dense_valid,
        "y_valid": y_valid,
        "preprocessor": preprocessor,
        "categorical_cardinalities": categorical_cardinalities,
        "selection_info": selection_info,
    }


# ---------------------------------------------------------------------------
# FT-Transformer model
# ---------------------------------------------------------------------------

class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular binary classification.

    Every categorical and numerical feature becomes its own token.
    Self-attention then learns interactions among those feature tokens.
    """

    def __init__(
        self,
        categorical_cardinalities: list[int],
        n_numeric: int,
        d_token: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ffn: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_categorical = len(
            categorical_cardinalities
        )

        self.n_numeric = n_numeric
        self.d_token = d_token

        # Each categorical feature gets its own embedding table.
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    cardinality,
                    d_token,
                )
                for cardinality
                in categorical_cardinalities
            ]
        )

        # Convert every numerical scalar into a learned token.
        self.numeric_weight = nn.Parameter(
            torch.randn(
                n_numeric,
                d_token,
            )
            * 0.02
        )

        self.numeric_bias = nn.Parameter(
            torch.zeros(
                n_numeric,
                d_token,
            )
        )

        # CLS token summarizes the entire transaction.
        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_token,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Convert the final CLS representation into one fraud logit.
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(
                d_token,
                1,
            ),
        )

    def forward(
        self,
        categorical: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = numeric.shape[0]

        tokens = []

        # Turn categorical values into feature tokens.
        if self.n_categorical:
            categorical_tokens = torch.stack(
                [
                    embedding(
                        categorical[:, i]
                    )
                    for i, embedding
                    in enumerate(
                        self.cat_embeddings
                    )
                ],
                dim=1,
            )

            tokens.append(
                categorical_tokens
            )

        # Turn every numerical value into a learned feature token.
        if self.n_numeric:
            numeric_tokens = (
                numeric.unsqueeze(-1)
                * self.numeric_weight.unsqueeze(0)
                + self.numeric_bias.unsqueeze(0)
            )

            tokens.append(
                numeric_tokens
            )

        if not tokens:
            raise ValueError(
                "FTTransformer received no input features."
            )

        feature_tokens = torch.cat(
            tokens,
            dim=1,
        )

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [
                cls,
                feature_tokens,
            ],
            dim=1,
        )

        x = self.transformer(x)

        # Position 0 is the CLS token.
        logits = self.head(
            x[:, 0]
        ).squeeze(1)

        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ft_transformer(
    data: dict,
    *,
    d_token: int = 64,
    n_heads: int = 8,
    n_layers: int = 3,
    d_ffn: int = 256,
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 4,
    seed: int = 42,
):
    """
    Train the FT-Transformer and keep the epoch with the highest
    validation AUPRC.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Training on: {device}"
    )

    model = FTTransformer(
        categorical_cardinalities=data[
            "categorical_cardinalities"
        ],
        n_numeric=data[
            "num_train"
        ].shape[1],
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        dropout=dropout,
    ).to(device)

    # Focal loss was the best-performing imbalance method
    # in Mohit's MLP baseline.
    criterion = FocalLoss(
        gamma=2.0,
        alpha=0.75,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    cat_train = torch.from_numpy(
        data["cat_train"]
    ).long()

    num_train = torch.from_numpy(
        data["num_train"]
    ).float()

    y_train = torch.from_numpy(
        data["y_train"]
    ).float()

    cat_valid = torch.from_numpy(
        data["cat_valid"]
    ).long()

    num_valid = torch.from_numpy(
        data["num_valid"]
    ).float()

    y_valid = data["y_valid"]

    best_auprc = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    history = []

    training_start = time.time()

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        permutation = torch.randperm(
            len(y_train)
        )

        running_loss = 0.0

        for start_idx in range(
            0,
            len(y_train),
            batch_size,
        ):
            idx = permutation[
                start_idx:
                start_idx + batch_size
            ]

            if len(idx) < 2:
                continue

            cat_batch = (
                cat_train[idx]
                .to(device)
            )

            num_batch = (
                num_train[idx]
                .to(device)
            )

            y_batch = (
                y_train[idx]
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(
                cat_batch,
                num_batch,
            )

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            # Prevent unstable Transformer gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * len(idx)
            )

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------

        model.eval()

        predictions = []

        with torch.no_grad():

            valid_batch_size = (
                batch_size * 4
            )

            for start_idx in range(
                0,
                len(y_valid),
                valid_batch_size,
            ):
                end_idx = (
                    start_idx
                    + valid_batch_size
                )

                cat_batch = (
                    cat_valid[
                        start_idx:end_idx
                    ]
                    .to(device)
                )

                num_batch = (
                    num_valid[
                        start_idx:end_idx
                    ]
                    .to(device)
                )

                logits = model(
                    cat_batch,
                    num_batch,
                )

                probabilities = (
                    torch.sigmoid(
                        logits
                    )
                    .cpu()
                    .numpy()
                )

                predictions.append(
                    probabilities
                )

        valid_scores = np.concatenate(
            predictions
        )

        valid_auprc = float(
            average_precision_score(
                y_valid,
                valid_scores,
            )
        )

        train_loss = (
            running_loss
            / len(y_train)
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_auprc": valid_auprc,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss {train_loss:.5f} | "
            f"valid AUPRC "
            f"{valid_auprc:.6f}"
        )

        # Save the best validation model.
        if valid_auprc > best_auprc:
            best_auprc = valid_auprc
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            bad_epochs = 0

        else:
            bad_epochs += 1

            if bad_epochs >= patience:
                print(
                    "Early stopping."
                )
                break

    if best_state is None:
        raise RuntimeError(
            "Training completed without "
            "producing a valid model."
        )

    model.load_state_dict(
        best_state
    )

    elapsed = (
        time.time()
        - training_start
    )

    print(
        f"\nBest validation AUPRC: "
        f"{best_auprc:.6f} "
        f"at epoch {best_epoch}"
    )

    print(
        f"Training time: "
        f"{elapsed / 60:.1f} minutes"
    )

    return model, {
        "best_auprc": best_auprc,
        "best_epoch": best_epoch,
        "train_seconds": elapsed,
        "history": history,
        "device": str(device),
        "d_token": d_token,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
    }

# ---------------------------------------------------------------------------
# Prediction and end-to-end run
# ---------------------------------------------------------------------------

def predict_ft_transformer(
    model: nn.Module,
    categorical: np.ndarray,
    numeric: np.ndarray,
    batch_size: int = 1024,
):
    """Return fraud probabilities for a processed dataset."""

    device = next(model.parameters()).device

    cat = torch.from_numpy(categorical).long()
    num = torch.from_numpy(numeric).float()

    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(num), batch_size):
            end = start + batch_size

            logits = model(
                cat[start:end].to(device),
                num[start:end].to(device),
            )

            predictions.append(
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

    return np.concatenate(predictions)


def run(
    data_path=None,
    split_dir=None,
    registry=None,
    report_dir=None,
    *,
    max_numeric: int = 100,
    seed: int = 42,
    **training_kwargs,
):
    """
    Run the complete Task #15 FT-Transformer experiment.

    The blind test split remains sealed.
    """

    # Load the team's canonical merged dataset.
    df = xb.load_merged(data_path)

    # Apply the same frozen chronological split used by the other models.
    split = xb.assign_split(
        df,
        split_dir,
    )

    train = df[
        split == "train"
    ].copy()

    valid = df[
        split == "valid"
    ].copy()

    split_source = split.attrs.get(
        "source",
        "unknown",
    )

    split_sizes = {
        name: int(
            (split == name).sum()
        )
        for name in xb.SPLIT_DAYS
    }

    del df

    # Reuse the same fe_v2 feature engineering as Mohit's MLP/XGBoost.
    train, [valid] = xb.add_features_v2(
        train,
        [valid],
    )

    # Select a Transformer-friendly feature subset and preprocess it.
    data = prepare_ft_data(
        train,
        valid,
        max_numeric=max_numeric,
    )

    print(
        "\nFT-Transformer feature set"
    )

    print(
        f"Categorical features: "
        f"{data['selection_info']['categorical_features']}"
    )

    print(
        f"Selected numeric features: "
        f"{data['selection_info']['numeric_features']}"
    )

    print(
        f"Original selected features: "
        f"{data['selection_info']['total_features']}"
    )

    print(
        f"Numerical tokens after missing flags: "
        f"{data['num_train'].shape[1]}"
    )

    # Train the model.
    model, training_info = train_ft_transformer(
        data,
        seed=seed,
        **training_kwargs,
    )

    # Re-score validation using the best saved epoch.
    valid_scores = predict_ft_transformer(
        model,
        data["cat_valid"],
        data["num_valid"],
    )

    # Use the same evaluation function as the team's other models.
    metrics = xb.evaluate(
        data["y_valid"],
        valid_scores,
        amount=valid[xb.AMOUNT_COL].to_numpy(),
    )

    print(
        "\nValidation metrics"
    )

    print(
        f"AUPRC:       {metrics['auprc']:.6f}"
    )

    print(
        f"ROC-AUC:     {metrics['roc_auc']:.6f}"
    )

    print(
        f"Recall@1%:   "
        f"{metrics['recall_at_1pct_review']:.6f}"
    )

    print(
        f"Recall@5%:   "
        f"{metrics['recall_at_5pct_review']:.6f}"
    )

    print(
        f"KS:          {metrics['ks']:.6f}"
    )

    record = {
        "model": "ft_transformer",
        "task": "Oct #15 DL baseline",
        "feature_set": "fe_v2_reduced",
        "split_id": split_source,
        "split_sizes": split_sizes,
        "seed": seed,
        "n_selected_features": (
            data[
                "selection_info"
            ][
                "total_features"
            ]
        ),
        "n_categorical": (
            data[
                "selection_info"
            ][
                "categorical_features"
            ]
        ),
        "n_numeric_selected": (
            data[
                "selection_info"
            ][
                "numeric_features"
            ]
        ),
        "n_numeric_tokens": int(
            data[
                "num_train"
            ].shape[1]
        ),
        "selected_columns": (
            data[
                "selection_info"
            ][
                "selected_columns"
            ]
        ),
        "params": training_info,
        "metrics": {
            "valid": metrics
        },
        "notes": (
            "FT-Transformer using Mohit's "
            "TabularPreprocessor and focal loss."
        ),
    }

    # Append the experiment to the shared results registry.
    registry_path = xb.log_run(
        record,
        registry,
    )

    print(
        f"\nLogged run to: "
        f"{registry_path}"
    )

    # Optionally save the model and validation predictions.
    if report_dir is not None:
        from pathlib import Path

        report_dir = Path(
            report_dir
        )

        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            report_dir
            / "ft_transformer.pt",
        )

        np.save(
            report_dir
            / "ft_transformer_valid_scores.npy",
            valid_scores,
        )

    return model, record, data

# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main():
    """Run Task #15 FT-Transformer from the command line."""

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="FT-Transformer baseline for October Task #15"
    )

    parser.add_argument("--data", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--report-dir", type=Path)

    parser.add_argument("--max-numeric", type=int, default=100)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ffn", type=int, default=256)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run(
        data_path=args.data,
        split_dir=args.split_dir,
        registry=args.registry,
        report_dir=args.report_dir,
        max_numeric=args.max_numeric,
        seed=args.seed,
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ffn=args.d_ffn,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()