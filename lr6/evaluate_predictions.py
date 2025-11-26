import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

df = pd.read_csv("lr6/predictions_log.csv")
# если true = -1 для некоторых — отбрось их
df = df[df["true_label"] != -1]

y_true = df["true_label"].astype(int)
y_pred = df["prediction"].astype(int)

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
print("Report:")
print(classification_report(y_true, y_pred))
