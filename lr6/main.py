import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import json
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from joblib import load


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

df = pd.read_csv(url)

# UCI сохраняет CSV через ; или через запятую — в этом датасете корректный разделитель ","
print(df.head())

df.to_csv("lr6/parkinsons_data.csv", index=False)

print("Файл parkinsons_data.csv успешно сохранён!")

# Загружаем датасет
df = pd.read_csv("lr6/parkinsons_data.csv")

X = df.drop(columns=["name", "status"])
y = df["status"]

# Фиксируем random_state для повторяемости
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

model = RandomForestClassifier(
    n_estimators=200, max_depth=12, random_state=random_state
)

model.fit(X_train, y_train)

dump(model, "lr6/rf_parkinson_model.joblib")

# --- ЛОГ ВОСПРОИЗВОДИМОСТИ ---
log = {
    "random_state": random_state,
    "model": "RandomForestClassifier",
    "parameters": model.get_params(),
    "sklearn_version": sklearn.__version__,
    "numpy_version": np.__version__,
}

with open("lr6/experiment_log.json", "w") as f:
    json.dump(log, f, indent=4)

print("Model trained and saved!")


model = load("lr6/rf_parkinson_model.joblib")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
