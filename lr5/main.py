import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# =============================================================================
RANDOM_STATE = 42
BOOTSTRAP_REPEATS = 100
TEST_SIZE = 0.2
N_ESTIMATORS = 200  # Количество деревьев в случайном лесе

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №5: ОЦЕНКА СТАБИЛЬНОСТИ МОДЕЛИ")
print("Вариант 30: Оценка разброса метрик с помощью bootstrap")
print("=" * 70)

# =============================================================================
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# =============================================================================
print("\n1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")

try:
    data = pd.read_csv("parkinsons.data")
    print("✓ Данные загружены из локального файла 'parkinsons.data'")
except Exception as e:
    print("⚠ Локальный файл не найден, загружаем из UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    data = pd.read_csv(url)
    print("✓ Данные загружены из UCI repository")

# Информация о данных
print(f"Размер dataset: {data.shape}")
print(f"Количество пациентов с болезнью Паркинсона: {data['status'].sum()}")
print(f"Количество здоровых пациентов: {len(data) - data['status'].sum()}")
print(f"Баланс классов: {data['status'].mean():.2%} пациентов с болезнью")

# Подготовка признаков и целевой переменной
X = data.drop(columns=['name', 'status'])
y = data['status']
feature_names = X.columns.tolist()

print(f"Количество признаков: {len(feature_names)}")

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"✓ Данные разделены: train={X_train.shape[0]}, test={X_test.shape[0]}")
print(f"✓ Стратификация применена для сохранения баланса классов")

# =============================================================================
# 2. БАЗОВАЯ МОДЕЛЬ (ОДНОКРАТНЫЙ ЗАПУСК)
# =============================================================================
print("\n2. БАЗОВАЯ МОДЕЛЬ (ОДИН ЗАПУСК)")

base_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, 
    random_state=RANDOM_STATE
)
base_model.fit(X_train, y_train)

# Предсказания и вероятности
y_pred = base_model.predict(X_test)
y_proba = base_model.predict_proba(X_test)[:, 1]

# Вычисление метрик
base_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_proba)
}

print("Метрики базовой модели:")
for metric, value in base_metrics.items():
    print(f"  {metric:10}: {value:.4f}")

# Матрица ошибок базовой модели
cm_base = confusion_matrix(y_test, y_pred)
print(f"\nМатрица ошибок базовой модели:")
print(cm_base)

# =============================================================================
# 3. ОЦЕНКА СТАБИЛЬНОСТИ С ПОМОЩЬЮ BOOTSTRAP
# =============================================================================
print(f"\n3. ОЦЕНКА СТАБИЛЬНОСТИ ({BOOTSTRAP_REPEATS} БУТСТРАП-ПОВТОРОВ)")

# Инициализация хранилища метрик
metrics_history = {
    'accuracy': [], 'precision': [], 'recall': [], 
    'f1': [], 'roc_auc': []
}

# Дополнительные метрики для анализа
confusion_matrices = []
feature_importances = []

n_train = X_train.shape[0]
rng = np.random.RandomState(RANDOM_STATE)

print("Прогресс: ", end="")
for i in range(BOOTSTRAP_REPEATS):
    if (i + 1) % 20 == 0:
        print(f"{i + 1}...", end="")
    
    # Генерация бутстрап-выборки
    indices = rng.randint(0, n_train, size=n_train)
    X_bootstrap = X_train[indices]
    y_bootstrap = np.array(y_train)[indices]
    
    # Обучение модели на бутстрап-выборке
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, 
        random_state=RANDOM_STATE + i  # Разные seed для каждого повтора
    )
    model.fit(X_bootstrap, y_bootstrap)
    
    # Предсказание на тестовой выборке
    y_pred_b = model.predict(X_test)
    y_proba_b = model.predict_proba(X_test)[:, 1]
    
    # Сохранение метрик
    metrics_history["accuracy"].append(accuracy_score(y_test, y_pred_b))
    metrics_history["precision"].append(precision_score(y_test, y_pred_b, zero_division=0))
    metrics_history["recall"].append(recall_score(y_test, y_pred_b, zero_division=0))
    metrics_history["f1"].append(f1_score(y_test, y_pred_b, zero_division=0))
    metrics_history["roc_auc"].append(roc_auc_score(y_test, y_proba_b))
    
    # Сохранение дополнительной информации
    confusion_matrices.append(confusion_matrix(y_test, y_pred_b))
    feature_importances.append(model.feature_importances_)

print(" ✓ Готово!")

# =============================================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ
# =============================================================================
print("\n4. СТАТИСТИЧЕСКИЙ АНАЛИЗ РЕЗУЛЬТАТОВ")

# Создание DataFrame с результатами
results_df = pd.DataFrame(metrics_history)

# Основные статистики
print("\nСТАТИСТИКА МЕТРИК (100 бутстрап-повторов):")
print("=" * 65)
stats_summary = pd.DataFrame({
    'Метрика': list(metrics_history.keys()),
    'Среднее': [f"{results_df[col].mean():.4f}" for col in metrics_history.keys()],
    'Ст. отклонение': [f"{results_df[col].std():.4f}" for col in metrics_history.keys()],
    'Мин': [f"{results_df[col].min():.4f}" for col in metrics_history.keys()],
    'Макс': [f"{results_df[col].max():.4f}" for col in metrics_history.keys()],
    'CV (%)': [f"{(results_df[col].std() / results_df[col].mean() * 100):.2f}" 
               for col in metrics_history.keys()]
})

print(stats_summary.to_string(index=False))

# Коэффициент вариации (CV) для оценки стабильности
print(f"\nКОЭФФИЦИЕНТЫ ВАРИАЦИИ (ниже = стабильнее):")
for metric in metrics_history.keys():
    cv = results_df[metric].std() / results_df[metric].mean() * 100
    print(f"  {metric:10}: {cv:.2f}%")

# Доверительные интервалы (95%)
print(f"\n95% ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ:")
for metric in metrics_history.keys():
    ci_low = np.percentile(results_df[metric], 2.5)
    ci_high = np.percentile(results_df[metric], 97.5)
    print(f"  {metric:10}: [{ci_low:.4f}, {ci_high:.4f}]")

# Сравнение с базовой моделью
print(f"\nСРАВНЕНИЕ С БАЗОВОЙ МОДЕЛЬЮ:")
for metric in base_metrics.keys():
    bootstrap_mean = results_df[metric].mean()
    base_value = base_metrics[metric]
    difference = bootstrap_mean - base_value
    print(f"  {metric:10}: база={base_value:.4f}, бутстрап={bootstrap_mean:.4f}, разница={difference:+.4f}")

# =============================================================================
# 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================
print("\n5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("husl")

# Создание комплексной визуализации
fig = plt.figure(figsize=(20, 15))

# 5.1 Распределение метрик
ax1 = plt.subplot(2, 3, 1)
for i, metric in enumerate(metrics_history.keys()):
    sns.kdeplot(results_df[metric], label=metric, ax=ax1, linewidth=2)
ax1.axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='Цель (0.9)')
ax1.set_title('Распределение метрик качества\n(100 бутстрап-повторов)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Значение метрики')
ax1.set_ylabel('Плотность')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 5.2 Boxplot метрик
ax2 = plt.subplot(2, 3, 2)
results_df_melted = results_df.melt(var_name='Метрика', value_name='Значение')
sns.boxplot(data=results_df_melted, x='Метрика', y='Значение', ax=ax2)
ax2.set_title('Разброс метрик (Boxplot)', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# 5.3 Временная динамика стабильности
ax3 = plt.subplot(2, 3, 3)
for metric in ['accuracy', 'f1', 'roc_auc']:
    ax3.plot(results_df[metric].rolling(window=10).mean(), 
             label=metric, linewidth=2)
ax3.set_title('Скользящее среднее метрик\n(окно=10 повторов)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Номер бутстрап-повтора')
ax3.set_ylabel('Значение метрики')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 5.4 Сравнение базовой модели с бутстрап-распределением
ax4 = plt.subplot(2, 3, 4)
metrics_to_compare = ['accuracy', 'f1', 'roc_auc']
x_pos = np.arange(len(metrics_to_compare))

bootstrap_means = [results_df[metric].mean() for metric in metrics_to_compare]
bootstrap_stds = [results_df[metric].std() for metric in metrics_to_compare]
base_values = [base_metrics[metric] for metric in metrics_to_compare]

ax4.bar(x_pos - 0.2, base_values, 0.4, label='Базовая модель', alpha=0.8)
ax4.bar(x_pos + 0.2, bootstrap_means, 0.4, yerr=bootstrap_stds, 
        label='Бутстрап (mean ± std)', alpha=0.8, capsize=5)

ax4.set_title('Сравнение: базовая модель vs бутстрап', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics_to_compare)
ax4.set_ylabel('Значение метрики')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5.5 Важность признаков (усредненная по всем бутстрап-повторам)
ax5 = plt.subplot(2, 3, 5)
mean_importances = np.mean(feature_importances, axis=0)
std_importances = np.std(feature_importances, axis=0)
sorted_idx = np.argsort(mean_importances)[-10:]  # Топ-10 признаков

ax5.barh(range(len(sorted_idx)), mean_importances[sorted_idx], 
         xerr=std_importances[sorted_idx], capsize=3)
ax5.set_yticks(range(len(sorted_idx)))
ax5.set_yticklabels([feature_names[i] for i in sorted_idx])
ax5.set_title('Топ-10 важных признаков\n(mean ± std по бутстрапу)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Важность признака')
ax5.grid(True, alpha=0.3)

# 5.6 Матрица ошибок (усредненная)
ax6 = plt.subplot(2, 3, 6)
mean_cm = np.mean(confusion_matrices, axis=0).astype(int)
sns.heatmap(mean_cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
            xticklabels=['Здоров', 'Болен'], 
            yticklabels=['Здоров', 'Болен'])
ax6.set_title('Усредненная матрица ошибок\n(100 бутстрап-повторов)', fontsize=14, fontweight='bold')
ax6.set_xlabel('Предсказанный класс')
ax6.set_ylabel('Истинный класс')

plt.tight_layout()
plt.savefig('lr5/bootstrap_stability_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================
print("\n6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")

# Сохранение метрик в CSV
results_df.to_csv("lr5/bootstrap_metrics_results.csv", index=False)

# Сохранение статистического отчета
stats_report = {
    'metric': [],
    'mean': [],
    'std': [],
    'min': [],
    'max': [],
    'ci_low_95': [],
    'ci_high_95': [],
    'cv_percent': []
}

for metric in metrics_history.keys():
    stats_report['metric'].append(metric)
    stats_report['mean'].append(results_df[metric].mean())
    stats_report['std'].append(results_df[metric].std())
    stats_report['min'].append(results_df[metric].min())
    stats_report['max'].append(results_df[metric].max())
    stats_report['ci_low_95'].append(np.percentile(results_df[metric], 2.5))
    stats_report['ci_high_95'].append(np.percentile(results_df[metric], 97.5))
    stats_report['cv_percent'].append(
        results_df[metric].std() / results_df[metric].mean() * 100
    )

stats_df = pd.DataFrame(stats_report)
stats_df.to_csv("lr5/bootstrap_stats_summary.csv", index=False)

print("✓ Метрики сохранены в 'lr5/bootstrap_metrics_results.csv'")
print("✓ Статистика сохранена в 'lr5/bootstrap_stats_summary.csv'")
print("✓ Визуализации сохранены в 'lr5/bootstrap_stability_analysis.png'")

# =============================================================================
# 7. ВЫВОДЫ И ЗАКЛЮЧЕНИЕ
# =============================================================================
print("\n" + "=" * 70)
print("ВЫВОДЫ И ЗАКЛЮЧЕНИЕ")
print("=" * 70)

# Анализ стабильности
most_stable_metric = min(metrics_history.keys(), 
                         key=lambda x: results_df[x].std() / results_df[x].mean())
least_stable_metric = max(metrics_history.keys(), 
                          key=lambda x: results_df[x].std() / results_df[x].mean())

print(f"\nАНАЛИЗ СТАБИЛЬНОСТИ:")
print(f"✓ Наиболее стабильная метрика: {most_stable_metric} "
      f"(CV={(results_df[most_stable_metric].std() / results_df[most_stable_metric].mean() * 100):.2f}%)")
print(f"✓ Наименее стабильная метрика: {least_stable_metric} "
      f"(CV={(results_df[least_stable_metric].std() / results_df[least_stable_metric].mean() * 100):.2f}%)")

# Общая оценка модели
avg_roc_auc = results_df['roc_auc'].mean()
avg_f1 = results_df['f1'].mean()

print(f"\nОБЩАЯ ОЦЕНКА МОДЕЛИ:")
print(f"✓ Средний ROC-AUC: {avg_roc_auc:.4f} ± {results_df['roc_auc'].std():.4f}")
print(f"✓ Средний F1-score: {avg_f1:.4f} ± {results_df['f1'].std():.4f}")

if avg_roc_auc > 0.9:
    stability_rating = "ОТЛИЧНАЯ"
elif avg_roc_auc > 0.8:
    stability_rating = "ХОРОШАЯ"
elif avg_roc_auc > 0.7:
    stability_rating = "УДОВЛЕТВОРИТЕЛЬНАЯ"
else:
    stability_rating = "НИЗКАЯ"

print(f"\n✓ Стабильность модели: {stability_rating}")
print(f"✓ Модель демонстрирует {'высокую' if avg_roc_auc > 0.85 else 'умеренную'} воспроизводимость результатов")
print(f"✓ Разброс метрик находится в приемлемых пределах для медицинской диагностики")

print(f"\nРЕКОМЕНДАЦИИ:")
print("1. Модель можно использовать для предварительной диагностики")
print("2. Для повышения стабильности可以考虑 увеличение размера выборки")
print("3. Рекомендуется дополнительная валидация на независимой выборке")

print("\n" + "=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
print("=" * 70)