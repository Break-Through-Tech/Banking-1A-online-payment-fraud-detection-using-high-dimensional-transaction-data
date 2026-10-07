from sklearn.metrics import (
average_precision_score,
precision_score,
recall_score,
f1_score,
confusion_matrix
roc_curve
)


def evaluate_model(y_true, y_scores):
"""
Calculate the main evaluation metrics for fraud detection.

y_true = actual fraud labels (0 or 1)
y_scores = model's fraud scores/probabilities
"""

# Turn scores into predictions
y_pred = (y_scores >= 0.5).astype(int)

# Main metrics
auprc = average_precision_score(y_true, y_scores)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# False positive rate
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
fpr = fp / (fp + tn)

# KS statistic
fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
ks = max(abs(tpr_curve - fpr_curve))

# Recall at 1% review rate
recall_1 = recall_at_review_rate(y_true, y_scores, 0.01)

# Recall at 5% review rate
recall_5 = recall_at_review_rate(y_true, y_scores, 0.05)

return {
"AUPRC": auprc,
"Precision": precision,
"Recall": recall,
"F1": f1,
"FPR": fpr,
"Recall@1%": recall_1,
"Recall@5%": recall_5,
"KS": ks
}


def recall_at_review_rate(y_true, y_scores, review_rate):
"""
Calculate how many fraud cases are found when reviewing
the highest-risk transactions.
"""

number_to_review = int(len(y_true) * review_rate)

# Sort transactions from highest fraud score to lowest
sorted_indices = y_scores.argsort()[::-1]

# Select the highest-risk transactions
top_transactions = sorted_indices[:number_to_review]

# Count fraud cases found
fraud_found = y_true[top_transactions].sum()
total_fraud = y_true.sum()

return fraud_found / total_fraud
