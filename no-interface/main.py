import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang, norm, uniform, f
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def generate_reliability_data(alpha, num_samples=1000):
    """Генерация данных согласно алгоритму из задания"""
    # 1. Генерация ξ ∈ [0,1]
    xi = np.random.uniform(0, 1, num_samples)
    
    # 2. Обратное преобразование для заданных распределений
    # F1: Равномерное [0,4]
    P1_base = uniform.ppf(xi, loc=0, scale=4)
    # F2: Эрланга (форма=4, масштаб=1)  
    P2_base = erlang.ppf(xi, 4, scale=1)
    # F3: Нормальное N(0,3)
    P3_base = norm.ppf(xi, loc=0, scale=3)
    
    # Расчет σ_i для каждого распределения
    sigma1 = np.sqrt(4**2 / 12)  # СКО равномерного [0,4]
    sigma2 = np.sqrt(4 * 1**2)   # СКО распределения Эрланга
    sigma3 = 3                   # СКО нормального распределения
    
    # Добавление погрешности ε_i ~ N(0, α*σ_i)
    epsilon1 = norm.rvs(loc=0, scale=alpha * sigma1, size=num_samples)
    epsilon2 = norm.rvs(loc=0, scale=alpha * sigma2, size=num_samples) 
    epsilon3 = norm.rvs(loc=0, scale=alpha * sigma3, size=num_samples)
    
    P1 = P1_base + epsilon1
    P2 = P2_base + epsilon2
    P3 = P3_base + epsilon3
    
    return P1, P2, P3

def create_proper_table(P1, P2, P3):
    """Создание таблицы в точном соответствии с форматом из задания"""
    # Первая строка: названия параметров
    table_data = [['Значение параметра', 'П_1', 'П_2', 'П_3']]
    
    # Последующие строки: номера измерений и значения
    for i in range(len(P1)):
        table_data.append([i+1, P1[i], P2[i], P3[i]])
    
    return table_data

def f_test_comparison(rss_simple, rss_complex, df_simple, df_complex, n_samples, alpha=0.05):
    """Критерий Фишера для сравнения моделей"""
    f_stat = ((rss_simple - rss_complex) / (df_complex - df_simple)) / (rss_complex / (n_samples - df_complex - 1))
    p_value = 1 - f.cdf(f_stat, df_complex - df_simple, n_samples - df_complex - 1)
    return f_stat, p_value, p_value < alpha

def select_polynomial_degree(X, y, max_degree=5):
    """Подбор порядка полинома на основе критерия Фишера"""
    n_samples = len(y)
    best_degree = 1
    models = {}
    rss_values = {}
    
    for degree in range(1, max_degree + 1):
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        model.fit(X, y)
        y_pred = model.predict(X)
        rss = np.sum((y - y_pred)**2)
        n_params = PolynomialFeatures(degree=degree).fit_transform(X).shape[1]
        
        models[degree] = model
        rss_values[degree] = (rss, n_params)
        
        print(f"Степень {degree}: RSS = {rss:.4f}, параметров = {n_params}")
    
    # Критерий Фишера: последовательное сравнение моделей
    for degree in range(2, max_degree + 1):
        rss_prev, df_prev = rss_values[degree-1]
        rss_curr, df_curr = rss_values[degree]
        
        f_stat, p_value, significant = f_test_comparison(
            rss_prev, rss_curr, df_prev, df_curr, n_samples
        )
        
        print(f"F-тест {degree-1} vs {degree}: F={f_stat:.4f}, p={p_value:.4f}, значимо: {significant}")
        
        if significant:
            best_degree = degree
        else:
            break
    
    return best_degree, models[best_degree]

# Основная программа
np.random.seed(42)
num_samples = 1000
alphas = [0.1, 0.5, 1.0, 1.5]

# Создаем фигуры для полей рассеяния
fig_scatter, axes_scatter = plt.subplots(len(alphas), 3, figsize=(18, 5*len(alphas)))
if len(alphas) == 1:
    axes_scatter = axes_scatter.reshape(1, -1)

results = []

for idx, alpha in enumerate(alphas):
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ ДЛЯ α = {alpha}")
    print(f"{'='*60}")
    
    # Генерация данных
    P1, P2, P3 = generate_reliability_data(alpha, num_samples)
    
    # Создание таблицы в требуемом формате
    table_data = create_proper_table(P1, P2, P3)
    print("Первые 5 строк таблицы:")
    for i in range(min(6, len(table_data))):
        print(table_data[i])
    
    # Построение ВСЕХ полей рассеяния
    pairs = [('P1', 'P2', P1, P2), ('P1', 'P3', P1, P3), ('P2', 'P3', P2, P3)]
    
    for col, (x_name, y_name, x_data, y_data) in enumerate(pairs):
        scatter = axes_scatter[idx, col].scatter(x_data, y_data, alpha=0.6, s=20)
        axes_scatter[idx, col].set_title(f'Поле рассеяния {x_name}-{y_name} (α={alpha})')
        axes_scatter[idx, col].set_xlabel(x_name)
        axes_scatter[idx, col].set_ylabel(y_name)
        axes_scatter[idx, col].grid(True, alpha=0.3)
    
    # Построение регрессионной модели П₃ = φ(П₁, П₂)
    X = np.column_stack((P1, P2))
    y = P3
    
    print(f"\nПодбор порядка полинома для α={alpha}:")
    best_degree, best_model = select_polynomial_degree(X, y, max_degree=4)
    
    # Оценка качества модели
    y_pred = best_model.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    
    results.append({
        'alpha': alpha,
        'best_degree': best_degree,
        'r2_score': r2,
        'model': best_model
    })
    
    print(f"Выбрана модель: полином {best_degree}-й степени, R² = {r2:.4f}")

# Визуализация всех графиков
plt.tight_layout()
plt.show()

# Итоговый отчет
print(f"\n{'='*60}")
print("ИТОГОВЫЙ ОТЧЕТ")
print(f"{'='*60}")
for result in results:
    print(f"α={result['alpha']}: полином {result['best_degree']}-й степени, R²={result['r2_score']:.4f}")

# Дополнительный анализ: сравнение влияния α на разброс данных
print(f"\n{'='*60}")
print("АНАЛИЗ ВЛИЯНИЯ ПОГРЕШНОСТИ")
print(f"{'='*60}")

for alpha in alphas:
    P1, P2, P3 = generate_reliability_data(alpha, num_samples)
    cov_matrix = np.cov(np.column_stack((P1, P2, P3)), rowvar=False)
    total_variance = np.trace(cov_matrix)
    print(f"α={alpha}: общая дисперсия = {total_variance:.4f}")