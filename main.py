import pandas as pd
import numpy as np


# Функция для расчета весов методом анализа иерархий (AHP)
def calculate_ahp(matrix_data, index_names):
    df = pd.DataFrame(matrix_data, index=index_names, columns=index_names)

    # 1. Расчет собственного вектора (Vj) - среднее геометрическое по строке
    # Формула: корень n-й степени из произведения элементов строки
    n = len(df)
    df["Vj (Собственный вектор)"] = df.prod(axis=1).pow(1 / n)

    # 2. Расчет вектора приоритетов (Pj) - нормализация Vj
    sum_vj = df["Vj (Собственный вектор)"].sum()
    df["Pj (Вес критерия)"] = df["Vj (Собственный вектор)"] / sum_vj

    return df, sum_vj


# --- ЧАСТЬ 1: Выбор облачной платформы (Пункт 4.1, Вариант 15) ---
print("=== РАСЧЕТ 4.1: ВЫБОР ОБЛАЧНОЙ ПЛАТФОРМЫ ===")
criteria_platform = [
    "Надежность и безопасность",
    "Аналитика данных",
    "Протоколы",
    "Инструменты визуализации",
]

# Матрица из задания 15 (4.1)
matrix_platform = [
    [1, 5, 7, 1 / 3],
    [1 / 5, 1, 1 / 3, 5],
    [1 / 7, 3, 1, 3],
    [3, 1 / 5, 1 / 3, 1],
]

df_platform, sum_vj_plat = calculate_ahp(matrix_platform, criteria_platform)

# Вывод результатов для платформы
print(df_platform[["Vj (Собственный вектор)", "Pj (Вес критерия)"]].round(4))
print(f"\nСумма Vj = {sum_vj_plat:.4f}")
print("-" * 50)


# --- ЧАСТЬ 2: Выбор протокола связи (Пункт 4.2, Вариант 15) ---
print("\n=== РАСЧЕТ 4.2: ВЫБОР ПРОТОКОЛА СВЯЗИ ===")
criteria_protocol = [
    "Скорость (Speed)",
    "Задержка (Delay)",
    "Пропускная способность (BW)",
    "Мощность (Power)",
]

# Матрица из задания 15 (4.2)
matrix_protocol = [
    [1, 1 / 5, 1 / 7, 1],
    [5, 1, 5, 7],
    [7, 1 / 5, 1, 3],
    [1, 1 / 7, 1 / 3, 1],
]

df_protocol, sum_vj_prot = calculate_ahp(matrix_protocol, criteria_protocol)

# Вывод весов критериев для протокола
print("Веса критериев (Pj):")
print(df_protocol[["Vj (Собственный вектор)", "Pj (Вес критерия)"]].round(4))
print(f"\nСумма Vj = {sum_vj_prot:.4f}")

# --- РАСЧЕТ ГЛОБАЛЬНЫХ ПРИОРИТЕТОВ ДЛЯ ПРОТОКОЛОВ ---
# Данные из методички (Таблицы 4-7) - веса альтернатив (Qij) для каждого критерия
# [cite: 59, 61, 63, 65]
alternatives = ["LTE Cat 0", "eMTC", "NB-IoT", "EC-GSM-IoT"]

# Qij взяты из методички (столбцы Qi из таблиц 4, 5, 6, 7)
# Порядок: [Speed, Delay, BW, Power]
q_values = np.array(
    [
        [0.36, 0.58, 0.59, 0.42],  # LTE Cat 0
        [0.36, 0.29, 0.28, 0.22],  # eMTC
        [0.09, 0.04, 0.06, 0.22],  # NB-IoT
        [0.18, 0.07, 0.05, 0.12],  # EC-GSM-IoT
    ]
)

# Вектор весов критериев Pj, который мы только что рассчитали
weights_pj = df_protocol["Pj (Вес критерия)"].values

# Расчет Ci = сумма (Qij * Pj)
global_priorities = np.dot(q_values, weights_pj)

results_df = pd.DataFrame(
    {"Альтернатива": alternatives, "Ci (Глобальный приоритет)": global_priorities}
)

print("\nИтоговый выбор протокола (Ci):")
print(results_df.sort_values(by="Ci (Глобальный приоритет)", ascending=False).round(4))
