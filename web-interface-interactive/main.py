import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import erlang, norm, uniform, f
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Настройка страницы
st.set_page_config(
    page_title="Анализ многомерных характеристик надежности",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация session_state
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"  # По умолчанию светлая тема
if 'font_size' not in st.session_state:
    st.session_state.font_size = "medium"
if 'results' not in st.session_state:
    st.session_state.results = None
if 'all_data' not in st.session_state:
    st.session_state.all_data = None

# Глобальная конфигурация размеров шрифта
font_configs = {
    "small": {
        "base": "14px",
        "h1": "28px",
        "h2": "22px", 
        "h3": "18px",
        "body": "14px",
        "small": "12px",
        "metric": "13px",
        "plot_title": 16,
        "plot_axis": 12,
        "plot_legend": 11
    },
    "medium": {
        "base": "16px",
        "h1": "32px",
        "h2": "24px",
        "h3": "20px", 
        "body": "16px",
        "small": "14px",
        "metric": "15px",
        "plot_title": 18,
        "plot_axis": 14,
        "plot_legend": 13
    },
    "large": {
        "base": "18px",
        "h1": "36px",
        "h2": "28px",
        "h3": "24px",
        "body": "18px", 
        "small": "16px",
        "metric": "17px",
        "plot_title": 20,
        "plot_axis": 16,
        "plot_legend": 15
    }
}

# Функции для управления темой и размером текста
def apply_custom_styles():
    """Применяет кастомные стили для темы и размера текста"""
    
    fs = font_configs[st.session_state.font_size]
    
    # Базовые стили для шрифтов
    base_css = f"""
    <style>
        /* Базовые настройки шрифтов */
        html, body, [class*="css"] {{
            font-size: {fs['base']} !important;
        }}
        
        /* Заголовки */
        h1 {{
            font-size: {fs['h1']} !important;
        }}
        h2 {{
            font-size: {fs['h2']} !important;
        }}
        h3 {{
            font-size: {fs['h3']} !important;
        }}
        
        /* Основной текст */
        .stMarkdown {{
            font-size: {fs['body']} !important;
        }}
        
        /* Мелкий текст */
        .stCaption, .stTooltip {{
            font-size: {fs['small']} !important;
        }}
        
        /* Метрики и карточки */
        .metric-card {{
            font-size: {fs['metric']} !important;
        }}
        
        /* Элементы форм */
        .stSelectbox, .stMultiselect, .stSlider, .stButton, .stTextInput {{
            font-size: {fs['body']} !important;
        }}
        
        /* Таблицы */
        .stDataFrame {{
            font-size: {fs['body']} !important;
        }}
        
        /* Вкладки */
        .stTabs {{
            font-size: {fs['body']} !important;
        }}
        
        /* ИСПРАВЛЕНИЕ: Цвета текста экспандеров для светлой темы */
        .main .streamlit-expanderHeader {{
            color: #31333F !important;
        }}
        
        .main .streamlit-expanderHeader:hover {{
            color: #2E86AB !important;
        }}
        
        /* Развернутый экспандер - белый текст на синем фоне */
        .main .streamlit-expanderHeader[aria-expanded="true"] {{
            color: #FFFFFF !important;
            background-color: #2E86AB !important;
        }}
        
        /* Иконки экспандеров */
        .main .streamlit-expanderIcon {{
            color: #31333F !important;
        }}
        
        .main .streamlit-expanderHeader[aria-expanded="true"] .streamlit-expanderIcon {{
            color: #FFFFFF !important;
        }}
    </style>
    """
    
    # Стили для темной темы
    dark_theme_css = f"""
    <style>
        /* Основные цвета темной темы */
        .main {{
            background-color: #0E1117;
            color: #FAFAFA;
        }}
        
        .stApp {{
            background-color: #0E1117;
        }}
        
        /* Сайдбар */
        .css-1d391kg {{
            background-color: #262730;
        }}
        
        /* Заголовки в сайдбаре - БЕЛЫЕ в темной теме */
        .sidebar-header {{
            color: #FAFAFA !important;
            font-size: {fs['h2']} !important;
        }}
        
        /* Текст */
        .stMarkdown {{
            color: #FAFAFA;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #FAFAFA;
        }}
        
        /* Вкладки - БЕЛЫЕ в темной теме */
        .stTabs [data-baseweb="tab"] {{
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: #333541 !important;
            color: #FAFAFA !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: #66B3FF !important;
            color: #0E1117 !important;
        }}
        
        /* Карточки и метрики */
        .result-card {{
            background-color: #262730;
            color: #FAFAFA;
            border-left: 5px solid #66B3FF;
        }}
        
        .instruction-box {{
            background-color: #262730;
            color: #FAFAFA;
            border-left: 5px solid #66B3FF;
        }}
        
        .tooltip {{
            background-color: #262730;
            color: #FAFAFA;
            border-left: 3px solid #66B3FF;
        }}
        
        /* Expanders - видимые в темной теме */
        .stExpander {{
            border: 1px solid #66B3FF !important;
            border-radius: 5px;
        }}
        
        .stExpander > div > div {{
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }}
        
        /* Элементы форм */
        .stSelectbox > div > div {{
            background-color: #262730;
            color: #FAFAFA;
        }}
        
        .stMultiselect > div > div {{
            background-color: #262730;
            color: #FAFAFA;
        }}
        
        .stTextInput > div > div > input {{
            background-color: #262730;
            color: #FAFAFA;
        }}
        
        .stSlider > div > div > div {{
            color: #FAFAFA;
        }}
        
        /* Метрики в темной теме */
        .stMetric {{
            color: #FAFAFA !important;
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: #FAFAFA !important;
        }}
        
        /* Экспандеры в темной теме */
        .main .streamlit-expanderHeader {{
            color: #FAFAFA !important;
        }}
        
        .main .streamlit-expanderHeader:hover {{
            color: #66B3FF !important;
        }}
        
        .main .streamlit-expanderHeader[aria-expanded="true"] {{
            color: #0E1117 !important;
            background-color: #66B3FF !important;
        }}
        
        .main .streamlit-expanderIcon {{
            color: #FAFAFA !important;
        }}
        
        .main .streamlit-expanderHeader[aria-expanded="true"] .streamlit-expanderIcon {{
            color: #0E1117 !important;
        }}
    </style>
    """
    
    # Стили для светлой темы
    light_theme_css = f"""
    <style>
        /* Основные цвета светлой темы */
        .main {{
            background-color: #FFFFFF;
            color: #31333F;
        }}
        
        .stApp {{
            background-color: #FFFFFF;
        }}
        
        /* Сайдбар - ТЕМНЫЙ в светлой теме */
        .css-1d391kg {{
            background-color: #262730;
        }}
        
        /* Заголовки в сайдбаре - БЕЛЫЕ в светлой теме (т.к. сайдбар темный) */
        .sidebar-header {{
            color: #FAFAFA !important;
            font-size: {fs['h2']} !important;
        }}
        
        /* Текст */
        .stMarkdown {{
            color: #31333F;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #31333F;
        }}
        
        /* Вкладки - черный текст в светлой теме */
        .stTabs [data-baseweb="tab"] {{
            background-color: #F0F2F6 !important;
            color: #31333F !important;
            font-weight: 600;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: #E6E9EF !important;
            color: #31333F !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: #2E86AB !important;
            color: #FFFFFF !important;
        }}
        
        /* Карточки и метрики */
        .result-card {{
            background-color: #F8F9FA;
            color: #31333F;
            border-left: 5px solid #2E86AB;
        }}
        
        .instruction-box {{
            background-color: #E8F4FD;
            color: #31333F;
            border-left: 5px solid #2E86AB;
        }}
        
        .tooltip {{
            background-color: #F0F2F6;
            color: #31333F;
            border-left: 3px solid #2E86AB;
        }}
        
        /* Expanders - базовые стили */
        .stExpander {{
            border: 1px solid #2E86AB !important;
            border-radius: 5px;
        }}
        
        .stExpander > div > div {{
            background-color: #FFFFFF !important;
            color: #31333F !important;
        }}
        
        /* Метрики в светлой теме */
        .stMetric {{
            color: #31333F !important;
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
            color: #31333F !important;
        }}
    </style>
    """
    
    # Применяем базовые стили шрифтов
    st.markdown(base_css, unsafe_allow_html=True)
    
    # Применяем тему
    if st.session_state.theme == "dark":
        st.markdown(dark_theme_css, unsafe_allow_html=True)
    else:
        st.markdown(light_theme_css, unsafe_allow_html=True)

# Остальные функции остаются без изменений
def generate_reliability_data(alpha, num_samples=1000, random_state=42):
    """Генерация данных согласно алгоритму из задания"""
    np.random.seed(random_state)
    
    # 1. Генерация ξ ∈ [0,1]
    xi = np.random.uniform(0, 1, num_samples)
    
    # 2. Обратное преобразование для заданных распределений
    P1_base = uniform.ppf(xi, loc=0, scale=4)          # Равномерное [0,4]
    P2_base = erlang.ppf(xi, 4, scale=1)               # Эрланга (форма=4, масштаб=1)
    P3_base = norm.ppf(xi, loc=0, scale=3)             # Нормальное N(0,3)
    
    # Расчет σ_i для каждого распределения
    sigma1 = np.sqrt(4**2 / 12)   # СКО равномерного [0,4]
    sigma2 = np.sqrt(4 * 1**2)    # СКО распределения Эрланга
    sigma3 = 3                    # СКО нормального распределения
    
    # Добавление погрешности ε_i ~ N(0, α*σ_i)
    epsilon1 = norm.rvs(loc=0, scale=alpha * sigma1, size=num_samples)
    epsilon2 = norm.rvs(loc=0, scale=alpha * sigma2, size=num_samples)
    epsilon3 = norm.rvs(loc=0, scale=alpha * sigma3, size=num_samples)
    
    P1 = P1_base + epsilon1
    P2 = P2_base + epsilon2
    P3 = P3_base + epsilon3
    
    return P1, P2, P3

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
    
    # Критерий Фишера: последовательное сравнение моделей
    for degree in range(2, max_degree + 1):
        rss_prev, df_prev = rss_values[degree-1]
        rss_curr, df_curr = rss_values[degree]
        
        f_stat, p_value, significant = f_test_comparison(
            rss_prev, rss_curr, df_prev, df_curr, n_samples
        )
        
        if significant:
            best_degree = degree
        else:
            break
    
    return best_degree, models[best_degree]

def create_scatter_plotly(P1, P2, P3, alpha, title):
    """Создание интерактивного поля рассеяния"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=P1, y=P2,
        mode='markers',
        marker=dict(
            size=8,
            color=P3,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="P3")
        ),
        hovertemplate=
        "<b>P1</b>: %{x:.3f}<br>" +
        "<b>P2</b>: %{y:.3f}<br>" +
        "<b>P3</b>: %{marker.color:.3f}<br>" +
        "<extra></extra>"
    ))
    
    # Используем темную тему для Plotly если выбрана темная тема, иначе светлую
    if st.session_state.theme == "dark":
        template = "plotly_dark"
        font_color = "white"
    else:
        template = "plotly_white"
        font_color = "black"
    
    # Получаем настройки шрифта для графиков
    fs = font_configs[st.session_state.font_size]
    
    fig.update_layout(
        title=f"{title} (α={alpha})",
        xaxis_title="P1",
        yaxis_title="P2",
        template=template,
        height=500,
        font=dict(
            size=fs['plot_axis'],
            color=font_color
        ),
        title_font=dict(
            size=fs['plot_title'],
            color=font_color
        ),
        paper_bgcolor='white' if st.session_state.theme == "light" else 'rgba(0,0,0,0)',
        plot_bgcolor='white' if st.session_state.theme == "light" else 'rgba(0,0,0,0)'
    )
    
    # Настраиваем цвета осей для светлой темы
    if st.session_state.theme == "light":
        fig.update_xaxes(
            linecolor='black',
            gridcolor='lightgray',
            tickfont=dict(color='black')
        )
        fig.update_yaxes(
            linecolor='black',
            gridcolor='lightgray', 
            tickfont=dict(color='black')
        )
    
    return fig

def create_3d_regression_plot(P1, P2, P3, model, alpha, degree):
    """Создание 3D визуализации регрессионной поверхности"""
    # Создаем сетку для предсказаний
    P1_range = np.linspace(P1.min(), P1.max(), 30)
    P2_range = np.linspace(P2.min(), P2.max(), 30)
    P1_grid, P2_grid = np.meshgrid(P1_range, P2_range)
    
    # Предсказания на сетке
    grid_points = np.column_stack((P1_grid.ravel(), P2_grid.ravel()))
    P3_pred_grid = model.predict(grid_points).reshape(P1_grid.shape)
    
    fig = go.Figure()
    
    # Добавляем исходные точки
    fig.add_trace(go.Scatter3d(
        x=P1, y=P2, z=P3,
        mode='markers',
        marker=dict(
            size=4,
            color=P3,
            colorscale='Viridis',
            opacity=0.7
        ),
        name='Исходные данные'
    ))
    
    # Добавляем регрессионную поверхность
    fig.add_trace(go.Surface(
        x=P1_grid, y=P2_grid, z=P3_pred_grid,
        colorscale='Plasma',
        opacity=0.7,
        name=f'Полином {degree}-й степени'
    ))
    
    # Используем темную тему для Plotly если выбрана темная тема, иначе светлую
    if st.session_state.theme == "dark":
        template = "plotly_dark"
        font_color = "white"
        bg_color = 'rgba(0,0,0,0)'
    else:
        template = "plotly_white"
        font_color = "black"
        bg_color = 'white'
    
    # Получаем настройки шрифта для графиков
    fs = font_configs[st.session_state.font_size]
    
    fig.update_layout(
        title=f'Регрессионная модель П₃ = φ(П₁, П₂) (α={alpha}, степень={degree})',
        scene=dict(
            xaxis_title='P1',
            yaxis_title='P2', 
            zaxis_title='P3',
            bgcolor=bg_color
        ),
        template=template,
        height=600,
        font=dict(
            size=fs['plot_axis'],
            color=font_color
        ),
        title_font=dict(
            size=fs['plot_title'],
            color=font_color
        ),
        paper_bgcolor=bg_color
    )
    
    # Настраиваем шрифты для осей в 3D сцене
    fig.update_scenes(
        xaxis=dict(
            title_font=dict(size=fs['plot_axis'], color=font_color),
            tickfont=dict(size=fs['plot_legend'], color=font_color)
        ),
        yaxis=dict(
            title_font=dict(size=fs['plot_axis'], color=font_color),
            tickfont=dict(size=fs['plot_legend'], color=font_color)
        ),
        zaxis=dict(
            title_font=dict(size=fs['plot_axis'], color=font_color),
            tickfont=dict(size=fs['plot_legend'], color=font_color)
        )
    )
    
    return fig

def compute_mean_curve(x, y, num_bins=20):
    """
    Вычисляет среднюю кривую через бининг и усреднение
    Возвращает x_bin_centers, y_means
    """
    # Создаем бины по x
    x_min, x_max = x.min(), x.max()
    bins = np.linspace(x_min, x_max, num_bins + 1)
    
    # Вычисляем средние значения в каждом бине
    x_bin_centers = (bins[:-1] + bins[1:]) / 2
    y_means = []
    
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(mask) > 0:
            y_means.append(np.mean(y[mask]))
        else:
            y_means.append(np.nan)
    
    # Убираем NaN значения
    valid_mask = ~np.isnan(y_means)
    return x_bin_centers[valid_mask], np.array(y_means)[valid_mask]

def add_mean_curves_to_scatter(fig, P1, P2, P3, row, col):
    """Добавляет средние кривые на графики рассеяния"""
    
    if col == 1:  # P1-P2 график
        # Средняя кривая P2 от P1
        x_curve, y_curve = compute_mean_curve(P1, P2)
        fig.add_trace(
            go.Scatter(
                x=x_curve, y=y_curve,
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=6, color='red'),
                name='Средняя P2 от P1',
                showlegend=False
            ), row=row, col=col
        )
        
    elif col == 2:  # P1-P3 график
        # Средняя кривая P3 от P1
        x_curve, y_curve = compute_mean_curve(P1, P3)
        fig.add_trace(
            go.Scatter(
                x=x_curve, y=y_curve,
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=6, color='red'),
                name='Средняя P3 от P1',
                showlegend=False
            ), row=row, col=col
        )
        
    elif col == 3:  # P2-P3 график
        # Средняя кривая P3 от P2
        x_curve, y_curve = compute_mean_curve(P2, P3)
        fig.add_trace(
            go.Scatter(
                x=x_curve, y=y_curve,
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=6, color='red'),
                name='Средняя P3 от P2',
                showlegend=False
            ), row=row, col=col
        )

def show_comprehensive_instructions():
    """Показывает полную инструкцию по работе с приложением"""
    
    with st.expander("🎯 Цель лабораторной работы", expanded=False):
        st.markdown("""
        <div class="instruction-box">
        <h3>🎯 Цель лабораторной работы</h3>
        <p>Построение многомерной регрессионной зависимости П₃ = φ(П₁, П₂) на основе законов распределения 
        трех параметров надежности с различной точностью регистрации данных.</p>
        
        <h3>🔧 Алгоритм работы</h3>
        <ol>
            <li><strong>Генерация случайных чисел</strong> ξ ∈ [0,1]</li>
            <li><strong>Вычисление параметров</strong> через обратное преобразование распределений</li>
            <li><strong>Добавление погрешности измерений</strong> ε_i ~ N(0, α·σ_i)</li>
            <li><strong>Построение полей рассеяния</strong> и регрессионной модели</li>
            <li><strong>Подбор порядка полинома</strong> по критерию Фишера</li>
        </ol>
        
        <h3>📊 Распределения параметров</h3>
        <ul>
            <li><strong>F₁(П)</strong> = Равномерное [0,4]</li>
            <li><strong>F₂(П)</strong> = Эрланга (форма=4, масштаб=1)</li>
            <li><strong>F₃(П)</strong> = Нормальное N(0,3)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🎛️ Настройка параметров эксперимента", expanded=False):
        st.markdown("""
        <div class="instruction-box">
        <h3>🎛️ Настройка параметров эксперимента</h3>
        
        <h4>Параметр α (уровень погрешности):</h4>
        <ul>
            <li><strong>α = 0</strong>: Идеальные измерения без погрешности</li>
            <li><strong>α = 0.1</strong>: Очень высокая точность измерений</li>
            <li><strong>α = 0.5</strong>: Средняя точность измерений</li>
            <li><strong>α = 1.0</strong>: Стандартная погрешность (равна СКО распределения)</li>
            <li><strong>α = 1.5</strong>: Высокая погрешность измерений</li>
        </ul>
        
        <h4>Количество прогонов:</h4>
        <ul>
            <li><strong>100-500</strong>: Быстрый расчет, подходит для тестирования</li>
            <li><strong>500-1000</strong>: Оптимальный баланс скорости и точности</li>
            <li><strong>1000-2000</strong>: Высокая точность, но дольше расчет</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("💡 Советы по использованию", expanded=False):
        st.markdown("""
        <div class="instruction-box">
        <h3>💡 Советы по использованию</h3>
        
        <h4>Рекомендации по настройке:</h4>
        <ul>
            <li><strong>Начните с α = [0.1, 0.5, 1.0]</strong> для сравнения разных уровней погрешности</li>
            <li><strong>Используйте 1000 прогонов</strong> для точных результатов</li>
            <li><strong>Для быстрого тестирования</strong> используйте 100-500 прогонов</li>
            <li><strong>Выбирайте несколько значений α</strong> для анализа влияния погрешности</li>
        </ul>
        
        <h4>Интерпретация результатов:</h4>
        <ul>
            <li><strong>R² (коэффициент детерминации)</strong> показывает качество модели (ближе к 1 = лучше)</li>
            <li><strong>Степень полинома</strong> показывает сложность регрессионной модели</li>
            <li><strong>Общая дисперсия</strong> характеризует разброс данных</li>
            <li><strong>Средние кривые</strong> на графиках показывают математическое ожидание зависимостей</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🧭 Навигация по разделам", expanded=False):
        st.markdown("""
        <div class="instruction-box">
        <h3>🧭 Навигация по разделам</h3>
        
        <h4>Основные вкладки:</h4>
        <ul>
            <li><strong>📈 Основные результаты</strong> - ключевые метрики и общий анализ влияния погрешности</li>
            <li><strong>🔄 Поля рассеяния</strong> - визуализация зависимостей между параметрами со средними кривыми</li>
            <li><strong>🧮 Регрессионные модели</strong> - 3D визуализация регрессионных поверхностей</li>
            <li><strong>📊 Исходные данные</strong> - таблицы данных, статистика и возможность скачивания</li>
            <li><strong>📖 Инструкция</strong> - настоящее руководство по работе с приложением</li>
        </ul>
        
        <h4>Особенности интерфейса:</h4>
        <ul>
            <li>Используйте <strong>настройки темы</strong> для комфортной работы при разном освещении</li>
            <li>Регулируйте <strong>размер текста</strong> для удобства чтения</li>
            <li>Все графики <strong>интерактивны</strong> - можно приближать, выделять области</li>
            <li>Данные можно <strong>скачать в CSV</strong> для дальнейшего анализа</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Основной интерфейс
def main():
    # Применяем кастомные стили в самом начале
    apply_custom_styles()
    
    # Настройки в сайдбаре
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Настройки интерфейса</div>', unsafe_allow_html=True)
    
    # Словарь для отображения размеров текста на русском
    font_size_options = {
        "Маленький": "small",
        "Средний": "medium", 
        "Большой": "large"
    }
    
    # Словарь для отображения тем на русском
    theme_options = {
        "светлая": "light",
        "тёмная": "dark"
    }
    
    # Выбор темы с русскими названиями
    theme_display = st.sidebar.radio(
        "🎨 Цветовая тема:",
        ["светлая", "тёмная"],
        index=1 if st.session_state.theme == "dark" else 0,
        help="Выберите светлую или темную тему интерфейса"
    )
    
    # Получаем внутреннее значение темы
    selected_theme = theme_options[theme_display]
    
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    # Выбор размера текста на русском языке
    font_size_display = st.sidebar.selectbox(
        "🔤 Размер текста:",
        ["Маленький", "Средний", "Большой"],
        index=["Маленький", "Средний", "Большой"].index(
            next(key for key, value in font_size_options.items() if value == st.session_state.font_size)
        ),
        help="Выберите удобный размер текста для чтения"
    )
    
    if font_size_options[font_size_display] != st.session_state.font_size:
        st.session_state.font_size = font_size_options[font_size_display]
        st.rerun()
    
    # Параметры эксперимента
    st.sidebar.markdown('<div class="sidebar-header">🧪 Параметры эксперимента</div>', unsafe_allow_html=True)
    
    num_samples = st.sidebar.slider(
        "Количество прогонов", 
        100, 2000, 1000, 100,
        help="Количество генерируемых точек данных. Больше прогонов = точнее результаты, но дольше расчет."
    )
    
    alphas = st.sidebar.multiselect(
        "Значения параметра α", 
        [0, 0.1, 0.5, 1.0, 1.5], 
        default=[0.1, 0.5, 1.0, 1.5],
        help="Уровень погрешности измерений. α=0 - без погрешности, α=1.0 - погрешность равна стандартному отклонению."
    )
    
    # Заголовок
    st.markdown('<h1 style="text-align: center; margin-bottom: 2rem;">📊 Анализ многомерных характеристик надежности</h1>', unsafe_allow_html=True)
    
    if st.sidebar.button("🚀 Запустить расчет", type="primary"):
        if not alphas:
            st.sidebar.error("⚠️ Пожалуйста, выберите хотя бы одно значение α")
        else:
            with st.spinner("Выполняются расчеты..."):
                results = []
                all_data = []
                
                progress_bar = st.progress(0)
                
                for i, alpha in enumerate(alphas):
                    # Генерация данных
                    P1, P2, P3 = generate_reliability_data(alpha, num_samples)
                    
                    # Сохранение данных
                    for j in range(num_samples):
                        all_data.append([alpha, j+1, P1[j], P2[j], P3[j]])
                    
                    # Построение регрессионной модели
                    X = np.column_stack((P1, P2))
                    y = P3
                    
                    best_degree, best_model = select_polynomial_degree(X, y, max_degree=5)
                    
                    # Оценка качества модели
                    y_pred = best_model.predict(X)
                    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                    
                    # Расчет общей дисперсии
                    cov_matrix = np.cov(np.column_stack((P1, P2, P3)), rowvar=False)
                    total_variance = np.trace(cov_matrix)
                    
                    results.append({
                        'alpha': alpha,
                        'best_degree': best_degree,
                        'r2_score': r2,
                        'total_variance': total_variance,
                        'model': best_model,
                        'P1': P1,
                        'P2': P2, 
                        'P3': P3
                    })
                    
                    progress_bar.progress((i + 1) / len(alphas))
                
                # Сохранение результатов в session_state
                st.session_state.results = results
                st.session_state.all_data = all_data
                
                st.success("✅ Расчеты завершены!")

    # Отображение результатов
    if st.session_state.results is not None:
        results = st.session_state.results
        all_data = st.session_state.all_data
        
        # Создаем DataFrame с результатами
        df_results = pd.DataFrame([{
            'α': r['alpha'],
            'Степень полинома': r['best_degree'], 
            'R²': r['r2_score'],
            'Общая дисперсия': r['total_variance']
        } for r in results])
        
        # Вкладки для разных разделов
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Основные результаты", "🔄 Поля рассеяния", "🧮 Регрессионные модели", "📊 Исходные данные", "📖 Инструкция"])
        
        with tab1:
            st.markdown('<h2 style="border-bottom: 2px solid; padding-bottom: 0.5rem; margin-top: 2rem;">Основные метрики</h2>', unsafe_allow_html=True)
            
            # Метрики в колонках
            cols = st.columns(len(results))
            for i, (col, result) in enumerate(zip(cols, results)):
                with col:
                    # Определяем цвет градиента в зависимости от темы
                    if st.session_state.theme == "dark":
                        gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                    else:
                        gradient = "linear-gradient(135deg, #1f77b4 0%, #2E86AB 100%)"
                    
                    st.markdown(f"""
                    <div style="background: {gradient}; color: white; padding: 1rem; border-radius: 10px; text-align: center; font-size: {font_configs[st.session_state.font_size]['metric']}px;">
                        <h3>α = {result['alpha']}</h3>
                        <h4>Степень: {result['best_degree']}</h4>
                        <p>R² = {result['r2_score']:.4f}</p>
                        <p>Дисперсия = {result['total_variance']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # График зависимости R² от α
            st.markdown('<h2 style="border-bottom: 2px solid; padding-bottom: 0.5rem; margin-top: 2rem;">Влияние погрешности на качество модели</h2>', unsafe_allow_html=True)
            
            # Используем темную тему для Plotly если выбрана темная тема, иначе светлую
            if st.session_state.theme == "dark":
                template = "plotly_dark"
                font_color = "white"
                bg_color = 'rgba(0,0,0,0)'
            else:
                template = "plotly_white"
                font_color = "black"
                bg_color = 'white'
            
            # Получаем настройки шрифта для графиков
            fs = font_configs[st.session_state.font_size]
            
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Scatter(
                x=[r['alpha'] for r in results],
                y=[r['r2_score'] for r in results],
                mode='lines+markers',
                marker=dict(size=12),
                line=dict(width=3)
            ))
            
            fig_r2.update_layout(
                title="Зависимость R² от уровня погрешности α",
                xaxis_title="α",
                yaxis_title="R²",
                template=template,
                height=400,
                font=dict(
                    size=fs['plot_axis'],
                    color=font_color
                ),
                title_font=dict(
                    size=fs['plot_title'],
                    color=font_color
                ),
                xaxis=dict(
                    title_font=dict(size=fs['plot_axis']),
                    tickfont=dict(size=fs['plot_legend'])
                ),
                yaxis=dict(
                    title_font=dict(size=fs['plot_axis']),
                    tickfont=dict(size=fs['plot_legend'])
                ),
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color
            )
            
            # Настраиваем цвета осей для светлой темы
            if st.session_state.theme == "light":
                fig_r2.update_xaxes(
                    linecolor='black',
                    gridcolor='lightgray',
                    tickfont=dict(color='black')
                )
                fig_r2.update_yaxes(
                    linecolor='black',
                    gridcolor='lightgray', 
                    tickfont=dict(color='black')
                )
            
            st.plotly_chart(fig_r2, use_container_width=True)
            
            # Анализ результатов
            st.markdown('<h2 style="border-bottom: 2px solid; padding-bottom: 0.5rem; margin-top: 2rem;">Анализ результатов</h2>', unsafe_allow_html=True)
            
            analysis_text = """
            **Наблюдаемые закономерности:**
            - С увеличением α (погрешности измерений) качество модели R² закономерно снижается
            - При малых α критерий Фишера выбирает более сложные модели (высокие степени полиномов)
            - При больших α выбираются простые модели для избежания переобучения
            - Общая дисперсия данных растет с увеличением α
            """
            
            st.markdown(analysis_text)
        
        with tab2:
            st.markdown('<h2 style="border-bottom: 2px solid; padding-bottom: 0.5rem; margin-top: 2rem;">Поля рассеяния параметров</h2>', unsafe_allow_html=True)
            st.markdown("""
            **Пояснение к графикам:**
            - **Красные линии с маркерами**: средние кривые, показывающие математическое ожидание одной переменной относительно другой
            - **Цвет точек**: отражает значение третьего параметра (используется цветовая шкала)
            - **Средняя кривая** вычисляется через разбиение на бины и усреднение значений в каждом бине
            """)
            
            for result in results:
                alpha = result['alpha']
                
                st.markdown(f"### Значение параметра α = {alpha}")
                
                # Создаем подграфики для всех пар параметров
                fig_scatter = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=[
                        f'P1-P2 (α={alpha}) - Средняя P2 от P1', 
                        f'P1-P3 (α={alpha}) - Средняя P3 от P1', 
                        f'P2-P3 (α={alpha}) - Средняя P3 от P2'
                    ]
                )
                
                # P1-P2
                fig_scatter.add_trace(
                    go.Scatter(
                        x=result['P1'], y=result['P2'],
                        mode='markers',
                        marker=dict(
                            size=5, 
                            color=result['P3'], 
                            colorscale='Viridis', 
                            showscale=False,
                            opacity=0.6
                        ),
                        name='P1-P2',
                        hovertemplate=
                        "<b>P1</b>: %{x:.3f}<br>" +
                        "<b>P2</b>: %{y:.3f}<br>" +
                        "<b>P3</b>: %{marker.color:.3f}<br>" +
                        "<extra></extra>"
                    ), row=1, col=1
                )
                
                # P1-P3  
                fig_scatter.add_trace(
                    go.Scatter(
                        x=result['P1'], y=result['P3'],
                        mode='markers', 
                        marker=dict(
                            size=5, 
                            color=result['P2'], 
                            colorscale='Plasma', 
                            showscale=False,
                            opacity=0.6
                        ),
                        name='P1-P3',
                        hovertemplate=
                        "<b>P1</b>: %{x:.3f}<br>" +
                        "<b>P3</b>: %{y:.3f}<br>" +
                        "<b>P2</b>: %{marker.color:.3f}<br>" +
                        "<extra></extra>"
                    ), row=1, col=2
                )
                
                # P2-P3
                fig_scatter.add_trace(
                    go.Scatter(
                        x=result['P2'], y=result['P3'],
                        mode='markers',
                        marker=dict(
                            size=5, 
                            color=result['P1'], 
                            colorscale='Rainbow', 
                            showscale=False,
                            opacity=0.6
                        ),
                        name='P2-P3',
                        hovertemplate=
                        "<b>P2</b>: %{x:.3f}<br>" +
                        "<b>P3</b>: %{y:.3f}<br>" +
                        "<b>P1</b>: %{marker.color:.3f}<br>" +
                        "<extra></extra>"
                    ), row=1, col=3
                )
                
                # Добавляем средние кривые на все графики
                add_mean_curves_to_scatter(
                    fig_scatter, 
                    result['P1'], result['P2'], result['P3'],
                    row=1, col=1
                )
                add_mean_curves_to_scatter(
                    fig_scatter, 
                    result['P1'], result['P2'], result['P3'],
                    row=1, col=2
                )
                add_mean_curves_to_scatter(
                    fig_scatter, 
                    result['P1'], result['P2'], result['P3'],
                    row=1, col=3
                )
                
                # Используем темную тему для Plotly если выбрана темная тема, иначе светлую
                if st.session_state.theme == "dark":
                    template = "plotly_dark"
                    font_color = "white"
                    bg_color = 'rgba(0,0,0,0)'
                else:
                    template = "plotly_white"
                    font_color = "black"
                    bg_color = 'white'
                
                # Получаем настройки шрифта для графиков
                fs = font_configs[st.session_state.font_size]
                
                fig_scatter.update_layout(
                    height=500, 
                    showlegend=False,
                    title_text=f"Поля рассеяния со средними кривыми (α={alpha})",
                    template=template,
                    font=dict(
                        size=fs['plot_axis'],
                        color=font_color
                    ),
                    title_font=dict(
                        size=fs['plot_title'],
                        color=font_color
                    ),
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color
                )
                
                # Обновляем шрифты для заголовков подграфиков
                fig_scatter.update_annotations(
                    font_size=fs['plot_axis'],
                    font_color=font_color
                )
                
                # Настраиваем подписи осей
                fig_scatter.update_xaxes(
                    title_text="P1", 
                    title_font=dict(size=fs['plot_axis'], color=font_color),
                    tickfont=dict(size=fs['plot_legend'], color=font_color)
                )
                fig_scatter.update_xaxes(
                    title_text="P1", 
                    title_font=dict(size=fs['plot_axis'], color=font_color),
                    tickfont=dict(size=fs['plot_legend'], color=font_color),
                    row=1, col=2
                )
                fig_scatter.update_xaxes(
                    title_text="P2", 
                    title_font=dict(size=fs['plot_axis'], color=font_color),
                    tickfont=dict(size=fs['plot_legend'], color=font_color),
                    row=1, col=3
                )
                fig_scatter.update_yaxes(
                    title_text="P2", 
                    title_font=dict(size=fs['plot_axis'], color=font_color),
                    tickfont=dict(size=fs['plot_legend'], color=font_color),
                    row=1, col=1
                )
                fig_scatter.update_yaxes(
                    title_text="P3", 
                    title_font=dict(size=fs['plot_axis'], color=font_color),
                    tickfont=dict(size=fs['plot_legend'], color=font_color),
                    row=1, col=2
                )
                fig_scatter.update_yaxes(
                    title_text="P3", 
                    title_font=dict(size=fs['plot_axis'], color=font_color),
                    tickfont=dict(size=fs['plot_legend'], color=font_color),
                    row=1, col=3
                )
                
                # Настраиваем цвета осей для светлой темы
                if st.session_state.theme == "light":
                    for i in range(1, 4):
                        fig_scatter.update_xaxes(
                            linecolor='black',
                            gridcolor='lightgray',
                            tickfont=dict(color='black'),
                            row=1, col=i
                        )
                        fig_scatter.update_yaxes(
                            linecolor='black',
                            gridcolor='lightgray', 
                            tickfont=dict(color='black'),
                            row=1, col=i
                        )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Добавляем пояснение для конкретного уровня α
                with st.expander(f"Интерпретация средних кривых для α = {alpha}"):
                    st.markdown(f"""
                    **Средние кривые показывают:**
                    
                    **График P1-P2:** Как в среднем изменяется P2 при изменении P1
                    **График P1-P3:** Как в среднем изменяется P3 при изменении P1  
                    **График P2-P3:** Как в среднем изменяется P3 при изменении P2
                    
                    **При α = {alpha}:**
                    - Средние кривые показывают основную тенденцию зависимости
                    - Разброс точек вокруг средней кривой характеризует влияние погрешности
                    - {'Четко видна зависимость между параметрами' if alpha <= 0.5 else 'Зависимость частично скрыта шумом'}
                    """)
        
        with tab3:
            st.markdown('<h2 style="border-bottom: 2px solid; padding-bottom: 0.5rem; margin-top: 2rem;">3D визуализация регрессионных моделей</h2>', unsafe_allow_html=True)
            
            for result in results:
                alpha = result['alpha']
                degree = result['best_degree']
                
                st.markdown(f"### Регрессионная модель для α = {alpha}")
                
                fig_3d = create_3d_regression_plot(
                    result['P1'], result['P2'], result['P3'],
                    result['model'], alpha, degree
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Информация о модели
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Степень полинома", degree)
                with col2:
                    st.metric("R²", f"{result['r2_score']:.4f}")
                with col3:
                    st.metric("Общая дисперсия", f"{result['total_variance']:.2f}")
                
                st.markdown("---")
        
        with tab4:
            st.markdown('<h2 style="border-bottom: 2px solid; padding-bottom: 0.5rem; margin-top: 2rem;">Исходные данные измерений</h2>', unsafe_allow_html=True)
            
            # Создаем DataFrame с данными
            df_data = pd.DataFrame(all_data, columns=['alpha', 'Номер', 'P1', 'P2', 'P3'])
            
            # Фильтр по alpha
            selected_alpha = st.selectbox("Выберите уровень α для просмотра данных:", alphas)
            
            df_filtered = df_data[df_data['alpha'] == selected_alpha].head(100)  # Показываем первые 100 строк
            
            st.dataframe(df_filtered, use_container_width=True)
            
            # Кнопка скачивания данных
            csv = df_data.to_csv(index=False)
            st.download_button(
                label="📥 Скачать все данные (CSV)",
                data=csv,
                file_name="reliability_data.csv",
                mime="text/csv"
            )
            
            # Статистика данных
            st.markdown("### Статистика данных")
            st.dataframe(df_data.groupby('alpha').agg({
                'P1': ['mean', 'std', 'min', 'max'],
                'P2': ['mean', 'std', 'min', 'max'], 
                'P3': ['mean', 'std', 'min', 'max']
            }).round(4))
        
        with tab5:
            show_comprehensive_instructions()
    
    else:
        # Если расчеты еще не проводились, показываем инструкцию на главной
        show_comprehensive_instructions()
        
        st.info("👈 Выберите параметры в боковой панели и нажмите 'Запустить расчет' для начала анализа")

if __name__ == "__main__":
    main()