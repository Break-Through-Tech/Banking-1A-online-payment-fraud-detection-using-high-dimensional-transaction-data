# FT-Transformer Baseline — October Task #15

Owner: Orator · Split `chrono-v1-c3eab22dc2` · Validation only · Blind test untouched

## Setup

- Architecture: FT-Transformer
- Shared preprocessing: `TabularPreprocessor` from the MLP baseline
- Feature set: reduced `fe_v2`
- Selected original features: 137
- Categorical features: 37
- Numerical features selected: 100
- Total Transformer tokens: 167
- Token dimension: 32
- Attention heads: 4
- Transformer layers: 3
- Feed-forward width: 128
- Dropout: 0.20
- Batch size: 128
- Learning rate: 2e-4
- Weight decay: 1e-5
- Loss: focal loss
- Best epoch: 8
- Hardware: Colab GPU

## Validation results

| Metric | Result |
|---|---:|
| AUPRC | **0.470634** |
| ROC-AUC | 0.883389 |
| Precision | 0.397422 |
| Recall | 0.513176 |
| F1 | 0.447942 |
| FPR | 0.027313 |
| Recall @ 1% review | 0.239945 |
| Recall @ 5% review | 0.540222 |
| KS | 0.616504 |
| Fraud-dollar capture @ 1% | 0.170846 |
| Fraud-dollar capture @ 5% | 0.505837 |

Training took about 18 minutes.

A second, more strongly regularized configuration reached 0.446720 AUPRC, so the first configuration was retained.

## Comparison

| Model | Validation AUPRC |
|---|---:|
| Tuned XGBoost | 0.682 |
| Tuned LightGBM | 0.668 |
| MLP, focal loss | 0.536 |
| **FT-Transformer** | **0.471** |

## Conclusion

The FT-Transformer provides the required second deep-learning baseline. Under this reduced-feature configuration, it trails the MLP and the tuned tree models. The result is retained for model comparison and possible ensemble analysis.
