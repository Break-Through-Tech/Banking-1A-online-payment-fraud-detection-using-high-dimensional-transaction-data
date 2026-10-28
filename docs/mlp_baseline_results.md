# Tabular MLP Baseline (October Task #14)

Owner: Mohit · Split `chrono-v1-c3eab22dc2` · Feature set `fe_v2` · Validation split only

**Result:** the best MLP (focal loss) reaches **validation AUPRC 0.536 ± 0.008** over 3 seeds. That is 0.146 below tuned XGBoost (0.682), so the MLP misses the "within 0.03" bar we set in the DL literature review. Adding it to XGBoost raises AUPRC by only 0.002, below the 0.005 ensemble bar. This matches the benchmark literature: on this kind of data, a plain MLP trails boosted trees.

## Model

| Part | Choice |
|---|---|
| Inputs | Same 445 fe_v2 features as XGBoost |
| Categorical | 37 columns (31 text fields + card1, card2, card3, card5, addr1, addr2) with learned embeddings, 315 dimensions in total. Vocabulary from train only. Values seen fewer than 20 times share a "rare" slot; missing has its own slot |
| Numeric | 408 columns, quantile-normalized to a normal distribution (fitted on 200k training rows), missing set to 0 |
| Missing flags | 53 flags, one per distinct missing pattern (the 339 V columns collapse into a few blocks) |
| Body | 3 hidden layers 512 → 256 → 128, BatchNorm, ReLU, dropout 0.25 (0.1 on embeddings). 663k parameters |
| Training | AdamW, lr 1e-3 with OneCycle schedule, batch 1,024, up to 15 epochs, early stopping on validation AUPRC (patience 3), best epoch kept |
| Hardware | CPU (1–2 threads), about 30 seconds per epoch, about 7 minutes per run |
| Code | `python -m fraud.models.mlp_baseline --loss focal` |

## Results

| Run | Imbalance handling | Seeds | AUPRC | ROC-AUC | Fraud caught, top 1% | Fraud caught, top 5% |
|---|---|---|---|---|---|---|
| **Focal loss** (γ = 2, α = 0.75) | Down-weights easy examples | 42, 43, 44 | **0.536 ± 0.008** | 0.906 | 25.2% | 60.1% |
| Weighted BCE, pos_weight 27.4 | Full class ratio | 42 | 0.511 | 0.903 | 24.5% | 57.3% |
| Weighted BCE, pos_weight 5.2 (√ratio), dropout 0.4 | Partial | 42 | 0.525 | 0.902 | 25.2% | 58.5% |
| *Tuned XGBoost (for reference)* | scale_pos_weight 15.7 | 42, 43, 44 | *0.682* | *0.935* | *28.2%* | *71.2%* |

Focal-loss seeds: 0.545, 0.534, 0.530.

What we learned:

- **Focal loss beat both weighted-BCE runs** by 0.01 to 0.03 AUPRC. Full class weighting was the worst option, which matches XGBoost, where the tuned weight was about half the full ratio.
- **The MLP overfits after about 10 epochs.** Training loss kept falling while validation AUPRC flattened between 0.51 and 0.55. Stronger dropout alone (0.4) didn't fix it.
- **The gap is mainly at the 5% review rate** (60% vs 71% of fraud caught). At 1% both are near the ceiling.

## Does it help the ensemble?

A quick check on validation (Leo's Task #17 will do this properly with a held-out blend):

| Blend of tuned XGBoost and MLP (focal, seed 42) | AUPRC |
|---|---|
| XGBoost alone | 0.6824 |
| Rank average, 10% MLP | 0.6628 |
| Log-odds average, 10% MLP | 0.6832 |
| Log-odds average, 30% MLP | 0.6844 |

Averaging log-odds adds at most 0.002 (and the weight was picked on validation, so even that is optimistic). Rank averaging hurts. **The MLP doesn't meet the bar to join the ensemble yet.**

## Recommendations

1. **Keep the MLP as the DL reference point**, but don't spend more CPU time on this architecture.
2. **Next MLP step: numerical embeddings** (periodic or piecewise-linear; see the DL literature review, section 1). That is the biggest documented gain for MLPs on tabular data. It needs a Colab GPU because it multiplies the input width.
3. **FT-Transformer (#15, Orator):** use the same preprocessing class (`TabularPreprocessor`) so the two DL models are comparable, and train on Colab GPU with reduced V columns.
4. **Customer history features (Emily, #16)** will probably help the MLP more than the trees, because the MLP can't build interactions such as "same card, new device" as easily.

## Files

- `src/fraud/models/mlp_baseline.py`, tests in `tests/test_mlp_baseline.py`
- `results/registry.jsonl`: five MLP runs (`task: "Oct #14 DL baseline"`), each with its epoch-by-epoch history
