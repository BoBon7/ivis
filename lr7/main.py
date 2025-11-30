import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from lifelines import KaplanMeierFitter


# -------- 1. Загрузка данных --------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
df = pd.read_csv(url, header=None, names=cols)

print("Размер данных:", df.shape)
print(df.head())

# -------- 2. Анализ нулевых значений --------
zero_na_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_counts = (df[zero_na_cols] == 0).sum()
zero_frac = zero_counts / len(df)

print("Количество нулей:\n", zero_counts)
print("Доля нулей:\n", zero_frac)

# -------- 3. Обработка пропусков (медиана по outcome) --------
df_imputed = df.copy()

for col in zero_na_cols:
    medians = df_imputed.groupby("Outcome")[col].transform("median")
    df_imputed[col] = df_imputed[col].mask(df_imputed[col] == 0, medians)

print("Готово! Обработка нулей выполнена.")

# Оставляем иммутированную таблицу
data = df_imputed.copy()

# -------- 4. Описательная статистика и графики --------
vars_to_plot = ["Glucose", "BMI", "Age", "BloodPressure", "Insulin"]

for var in vars_to_plot:
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=data,
        x=var,
        hue="Outcome",
        kde=False,
        bins=30,
        element="step",
        stat="count",
    )
    plt.title(f"Распределение признака {var} для групп: без диабета и с диабетом")
    plt.xlabel(f"Значение показателя {var}")
    plt.ylabel("Количество наблюдений")
    plt.tight_layout()
    plt.show()
    # Логи-пояснения
    print(f"\n[ЛОГ] Гистограмма показателя {var}:")
    print(
        f"- Признак {var} показывает распределение значений по двум группам (Outcome 0 и 1)."
    )
    print("- Цвет показывает пациентов без диабета (0) и с диабетом (1).")
    print(f"- График помогает увидеть, отличаются ли группы по {var}.")
    print(
        "- Если распределения двух групп заметно различаются, признак может быть важным для классификации.\n"
    )


for var in vars_to_plot:
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Outcome", y=var, data=data)
    plt.title(f"Распределение {var} в группах (0 — без диабета, 1 — с диабетом)")
    plt.xlabel("Наличие диабета (Outcome)")
    plt.ylabel(f"Значения показателя {var}")
    plt.figtext(
        0.5,
        -0.05,
        "Boxplot показывает медиану (линия), межквартильный интервал (IQR) и возможные выбросы.",
        ha="center",
    )
    plt.tight_layout()
    plt.show()
    # Детализированные логи
    print(f"\n[ЛОГ] Boxplot признака {var}:")
    print(
        f"- На графике видно, как распределён признак {var} у людей с диабетом и без."
    )
    print("- Центральная линия внутри ящика — медиана.")
    print(
        "- Высота 'ящика' — межквартильный размах (IQR), показывающий вариацию в середине распределения."
    )
    print("- Точки вне усов — потенциальные выбросы.")
    print(
        "- Если медианы двух групп различаются заметно → этот признак отличает классы и важен для анализа.\n"
    )

# Табличная статистика
desc_by_outcome = data.groupby("Outcome")[vars_to_plot].describe()

print("Описательная статистика:")
print(desc_by_outcome)


# -------- 5. Корреляция Spearman --------
def spearmanr_pval_matrix(df_num):
    cols = df_num.columns
    rho = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    pval = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            r, p = stats.spearmanr(df_num[cols[i]], df_num[cols[j]])
            rho.iloc[i, j] = r
            pval.iloc[i, j] = p
    return rho, pval


numeric_cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
rho, pvals = spearmanr_pval_matrix(data[numeric_cols])

print("Матрица корреляций Spearman:")
print(rho)

print("P-values Spearman:")
print(pvals)

# Специально: корреляция беременностей
preg_corr = rho["Pregnancies"].sort_values(key=np.abs, ascending=False)
print("Корреляции Pregnancies:")
print(preg_corr)

# -------- 6. Модели и ROC-кривые --------
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "Логистическая регрессия": LogisticRegression(max_iter=1000, solver="liblinear"),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(probability=True, kernel="rbf", C=1.0),
}

plt.figure(figsize=(8, 6))
metrics_list = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    y_pred = (y_proba >= 0.5).astype(int)
    metrics_list.append(
        {
            "Модель": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": auc,
        }
    )

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC-кривые для моделей")
plt.xlabel("Доля ложноположительных")
plt.ylabel("Доля истинноположительных")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
print("\n[ЛОГ] ROC-анализ моделей:")
print("- ROC-кривые показывают баланс между True Positive Rate и False Positive Rate.")
print("- Чем выше кривая и ближе к верхнему левому углу, тем лучше модель.")
print("- AUC (Area Under Curve) — итоговая мера качества модели: чем выше, тем лучше.")
print(
    "- В логах ниже вы увидите сравнение: Accuracy, Precision, Recall, F1, ROC-AUC.\n"
)


metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

# -------- 7. PCA --------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

plt.figure(figsize=(7, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7, edgecolor="k")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("PCA: отображение данных в двух компонентах")
plt.colorbar(label="Outcome (наличие диабета)")
plt.tight_layout()
plt.show()
print("\n[ЛОГ] PCA анализ:")
print("- Данные сжаты до двух главных компонент без потери структуры.")
print("- Точки на графике — пациенты, цвет — наличие диабета.")
print(
    "- Если видны раздельные облака → признак Outcome хорошо объясняется комбинацией исходных признаков."
)
print(f"- Доля объяснённой дисперсии: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}\n")


pca_df = pd.DataFrame({"PC": ["PC1", "PC2"], "Explained_variance": explained})
print(pca_df)

# -------- 8. Kaplan-Meier --------
kmf = KaplanMeierFitter()
T = data["Age"]
E = data["Outcome"]

kmf.fit(T, E, label="Все пациенты")
kmf.plot_survival_function()
plt.title("Кривая выживаемости Kaplan-Meier (время = возраст)")
plt.xlabel("Возраст")
plt.ylabel("Вероятность отсутствия диабета")
plt.tight_layout()
plt.show()
print("\n[ЛОГ] Survival-анализ (Kaplan-Meier):")
print(
    "- Кривая показывает вероятность 'выживания' (то есть НЕ иметь диабет) в зависимости от возраста."
)
print("- Падение кривой означает рост доли пациентов с диагнозом к большему возрасту.")
print("- Это НЕ медицинский прогноз, а иллюстративная аналитика по данному датасету.\n")


# -------- 9. Сохранение всех результатов в Excel --------
with pd.ExcelWriter("lr7/results_summary.xlsx") as writer:
    desc_by_outcome.to_excel(writer, sheet_name="Описательная статистика")
    rho.to_excel(writer, sheet_name="Корреляции Spearman")
    pvals.to_excel(writer, sheet_name="P-values Spearman")
    metrics_df.to_excel(writer, sheet_name="Метрики моделей")
    pca_df.to_excel(writer, sheet_name="PCA")

print("Файл results_summary.xlsx успешно сохранён!")
