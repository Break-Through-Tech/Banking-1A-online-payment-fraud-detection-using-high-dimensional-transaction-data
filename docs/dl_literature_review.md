# Deep Learning for Tabular Fraud: Literature Refresh

October Task #9 · Owner: Mohit · Blocks #14 (MLP) and #15 (second DL model)

The question: which deep learning architectures should the team build in October, given our data (590,540 transactions, 432 mostly anonymized features, 3.5% fraud, heavy missingness), our compute (laptops, Colab GPU) and the September Task #15 result that a customer ID can be rebuilt.

## 1. What the benchmarks say

**Trees are still the bar.** Two large comparisons reach the same conclusion:

- Shwartz-Ziv and Armon (2021) re-ran recent deep tabular models on their own papers' datasets. XGBoost outperformed them and needed much less tuning. An **ensemble of deep models and XGBoost beat XGBoost alone**.
- Grinsztajn et al. (NeurIPS 2022), 45 datasets with full hyperparameter search, found tree models remain state of the art on medium-sized data. Their three reasons map directly onto our data:

| Weakness of neural nets (Grinsztajn et al.) | Why it hits our dataset |
|---|---|
| Not robust to uninformative features | Many of the 339 V columns are near-duplicates or noise |
| Don't preserve the orientation of the data (axis-aligned splits) | Missing-value blocks and match flags are axis-aligned signals |
| Struggle to learn irregular functions | Fraud rate jumps between specific card, email and device values |

**But a well-built MLP is a strong baseline.** Gorishniy et al. (NeurIPS 2021) proposed a ResNet-style MLP and FT-Transformer (Feature Tokenizer + Transformer). They found no universal winner between deep models and GBDT, and said the field lacked good simple baselines. Their follow-up (NeurIPS 2022) showed that **embedding numerical features** (piecewise-linear or periodic embeddings, not raw scalars) moved a plain MLP from last place to on par with the Transformer and close to CatBoost and XGBoost in average rank.

## 2. The four candidate architectures

| | MLP (+ embeddings) | TabNet | FT-Transformer | SAINT |
|---|---|---|---|---|
| Idea | Feed-forward net; categorical embeddings; optional numerical embeddings | Sequential attention picks a subset of features per step (Arik and Pfister, AAAI 2021) | Every feature becomes a token; Transformer attends across features | Attention across features **and across rows in a batch**, plus contrastive pre-training (Somepalli et al. 2021) |
| Strength | Fast, simple, strong with good preprocessing | Built-in, per-transaction feature masks (interpretability) | Strongest deep model in the Gorishniy benchmarks | Claims to beat boosted trees on average |
| Cost on our data | Seconds to minutes per epoch on CPU | Slow. On IEEE-CIS a recent study measured about 1,650 s per fold vs 25 s for LightGBM | Attention over about 445 feature tokens per row; needs a GPU and fewer features | Row attention depends on batch composition; heaviest to train and to serve |
| IEEE-CIS evidence | No direct benchmark, but its entity embeddings fed to LightGBM gave the best AUC-ROC (0.961) in that study | Off-the-shelf TabNet trailed tree pipelines (AUC-ROC 0.952 vs 0.961) and degraded sharply on 10% of the data | None found | None found |
| Fit for a real-time fraud score | Good | Fair | Fair with GPU | Poor (score depends on other rows in the batch) |

The IEEE-CIS study is Interpretable versus Learned Encoders for High-Cardinality Fraud Detection (2026). It used random cross-validation, not a time split, so its absolute numbers aren't comparable to ours. Its ranking is still useful.

## 3. What our own data adds

- **The customer ID rebuild works** (Sept Task #15): the key card1 + addr1 + (day − D1) matches the D3 gap 95% of the time. That makes **customer-level aggregates** (Emily, Oct #16) and, later, **sequence models** possible. Only 50% of validation rows have training history, so any model must handle "no history".
- **Missingness carries signal** (missing addr1 means 4.5 times the fraud rate). Neural nets need explicit missing-value flags; trees get this for free.
- **High-cardinality categories** (card1 with 12,295 values, DeviceInfo with 1,556) suit learned embeddings with a minimum-count cut-off.
- **Imbalance:** about 1 fraud per 29 legitimate. The challenge brief suggests class weighting and focal loss.

## 4. Decision

1. **Task #14 (Mohit): MLP with embeddings.** Categorical embeddings (vocabulary learned on train, rare values pooled), quantile-normalized numerics, missing-value flags for every numeric column with gaps, `BCEWithLogitsLoss` with `pos_weight`, focal loss as a variant, early stopping on validation AUPRC. This is the first DL entry and the yardstick for everything after it.
2. **Task #15 (Orator): FT-Transformer**, not TabNet, as the second model. It's the strongest deep model in the benchmarks. TabNet already has a negative IEEE-CIS result. To make attention affordable, train on Colab GPU with a reduced feature set (V columns cut by missingness block and correlation, as in the literature review section 1.3), about 120 to 150 tokens. TabNet stays the fallback if FT-Transformer doesn't fit in memory or time.
3. **Not this month: SAINT.** Its row attention makes a transaction's score depend on the other transactions in the batch, which is hard to justify for production fraud scoring, and it's the most expensive option.
4. **November stretch: a sequence model** (GRU over each rebuilt customer's earlier transactions), only if Oct #16 aggregates show customer history helps. It must use past transactions only.
5. **Use DL where it adds the most: the ensemble** (Oct #17). Benchmarks agree that deep models add the most value when combined with boosted trees.

## 5. Success criteria for the DL baselines

- Same split (`chrono-v1-c3eab22dc2`), same metrics, logged to the results registry.
- A DL model is **worth keeping if it is within 0.03 AUPRC of tuned XGBoost or improves the ensemble** by at least 0.005 AUPRC on validation. We don't expect it to beat the trees alone.
- Report training time and hardware, because real-time scoring cost matters to the business case.

## Sources

- Shwartz-Ziv, R. and Armon, A. *Tabular Data: Deep Learning is Not All You Need.* Information Fusion, 2022. https://arxiv.org/abs/2106.03253
- Grinsztajn, L., Oyallon, E. and Varoquaux, G. *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 2022 Datasets and Benchmarks. https://arxiv.org/abs/2207.08815
- Gorishniy, Y. et al. *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021. https://arxiv.org/abs/2106.11959
- Gorishniy, Y., Rubachev, I. and Babenko, A. *On Embeddings for Numerical Features in Tabular Deep Learning.* NeurIPS 2022. https://arxiv.org/abs/2203.05556 (summary: https://research.yandex.com/blog/embeddings-for-numerical-features-in-tabular-deep-learning)
- Arik, S. O. and Pfister, T. *TabNet: Attentive Interpretable Tabular Learning.* AAAI 2021. https://ojs.aaai.org/index.php/AAAI/article/view/16826
- Somepalli, G. et al. *SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training.* 2021. https://arxiv.org/abs/2106.01342
- *Interpretable versus Learned Encoders for High-Cardinality Fraud Detection* (IEEE-CIS). 2026. https://arxiv.org/html/2607.00477
