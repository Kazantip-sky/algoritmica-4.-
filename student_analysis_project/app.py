from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import base64
import json
from matplotlib.figure import Figure

app = Flask(__name__)

# Глобальные переменные для хранения данных
df = None
cleaned_hypothesis_data = None
group_with_courses = None
group_without_courses = None

def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    global df, cleaned_hypothesis_data, group_with_courses, group_without_courses
    
    # Загрузка данных
    df = pd.read_csv('StudentsPerformance.csv')
    
    # Подготовка данных
    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['average_score'] = df['total_score'] / 3
    df['target_group'] = df['average_score'] >= 60
    
    # Подготовка данных для гипотезы
    bachelor_degree = "bachelor's degree"
    master_degree = "master's degree"
    higher_education = [bachelor_degree, master_degree]
    
    hypothesis_data = df.copy()
    hypothesis_data['has_higher_edu_parents'] = hypothesis_data['parental level of education'].isin(higher_education)
    hypothesis_data['took_prep_course'] = hypothesis_data['test preparation course'] == 'completed'
    
    # Очистка данных
    hypothesis_data = hypothesis_data[
        (hypothesis_data['math score'] > 0) & 
        (hypothesis_data['math score'] < 100) &
        (hypothesis_data['reading score'] > 0) & 
        (hypothesis_data['reading score'] < 100) &
        (hypothesis_data['writing score'] > 0) & 
        (hypothesis_data['writing score'] < 100)
    ]
    
    cleaned_hypothesis_data = hypothesis_data[hypothesis_data['has_higher_edu_parents'] == False].copy()
    group_with_courses = cleaned_hypothesis_data[cleaned_hypothesis_data['took_prep_course'] == True]
    group_without_courses = cleaned_hypothesis_data[cleaned_hypothesis_data['took_prep_course'] == False]

def create_base64_plot(fig):
    """Создание base64 изображения из matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

def generate_dashboard_data():
    """Генерация данных для дашборда"""
    target_students = df[df['target_group']]
    total_students = len(df)
    
    # Основные метрики
    metrics = {
        'total_students': total_students,
        'target_students': len(target_students),
        'target_percentage': round(len(target_students) / total_students * 100, 1),
        'non_target_students': total_students - len(target_students),
        'non_target_percentage': round((total_students - len(target_students)) / total_students * 100, 1),
        'avg_math_score': round(df['math score'].mean(), 1),
        'avg_reading_score': round(df['reading score'].mean(), 1),
        'avg_writing_score': round(df['writing score'].mean(), 1),
        'avg_total_score': round(df['average_score'].mean(), 1)
    }
    
    # Распределение по полу
    gender_dist = df['gender'].value_counts().to_dict()
    
    # Распределение по курсам подготовки
    course_dist = df['test preparation course'].value_counts().to_dict()
    
    # Распределение по образованию родителей
    education_dist = df['parental level of education'].value_counts().to_dict()
    
    # Для гипотезы
    non_higher_ed_count = len(df[~df['parental level of education'].isin(["bachelor's degree", "master's degree"])])
    
    return {
        'metrics': metrics,
        'gender_dist': gender_dist,
        'course_dist': course_dist,
        'education_dist': education_dist,
        'non_higher_ed_percentage': round(non_higher_ed_count / total_students * 100, 1)
    }

def generate_ideas_data():
    """Генерация данных для страницы идей"""
    
    non_higher_ed_count = len(df[~df['parental level of education'].isin(["bachelor's degree", "master's degree"])])
    non_higher_ed_percent = non_higher_ed_count / len(df) * 100
    
    free_lunch_avg = df[df['lunch'] == 'free/reduced']['average_score'].mean()
    standard_lunch_avg = df[df['lunch'] == 'standard']['average_score'].mean()
    
    group_a_avg = df[df['race/ethnicity'] == 'group A']['average_score'].mean()
    group_e_avg = df[df['race/ethnicity'] == 'group E']['average_score'].mean()
    
    ideas = [
        {
            "id": 1,
            "name": "Интенсивные онлайн-курсы по математике",
            "target": "Абитуриенты со слабой математической подготовкой",
            "rationale": f"Средний балл по математике: {df['math score'].mean():.1f}, что ниже чем по чтению ({df['reading score'].mean():.1f}) и письму ({df['writing score'].mean():.1f})",
            "potential_market": round((len(df[df['math score'] < 60]) / len(df) * 100), 1),
            "key_metric": "math_score_below_60"
        },
        {
            "id": 2,
            "name": "Персонализированные курсы для детей из семей без высшего образования",
            "target": "Семьи где родители имеют среднее или неполное высшее образование",
            "rationale": f"Абитуриенты из таких семей составляют {non_higher_ed_percent:.1f}% от общего числа",
            "potential_market": round(non_higher_ed_percent, 1),
            "key_metric": "non_higher_ed_students"
        },
        {
            "id": 3,
            "name": "Программа 'Обед + Уроки'",
            "target": "Абитуриенты с бесплатным/льготным питанием",
            "rationale": f"Средний балл у абитуриентов с бесплатным питанием: {free_lunch_avg:.1f}, у остальных: {standard_lunch_avg:.1f}",
            "potential_market": round((len(df[df['lunch'] == 'free/reduced']) / len(df) * 100), 1),
            "key_metric": "free_lunch_students"
        },
        {
            "id": 4,
            "name": "Подготовительные курсы с фокусом на письмо",
            "target": "Абитуриенты, которым сложно дается письменная часть",
            "rationale": f"Средний балл по письму: {df['writing score'].mean():.1f}, минимальный: {df['writing score'].min()}, максимальный: {df['writing score'].max()}",
            "potential_market": round((len(df[df['writing score'] < 60]) / len(df) * 100), 1),
            "key_metric": "writing_score_below_60"
        },
        {
            "id": 5,
            "name": "Групповые занятия по этническим группам",
            "target": "Определенные этнические группы с низкими результатами",
            "rationale": f"Разница в средних баллах между группами: Group A: {group_a_avg:.1f}, Group E: {group_e_avg:.1f}",
            "potential_market": round((len(df[df['race/ethnicity'] == 'group A']) / len(df) * 100), 1),
            "key_metric": "group_a_students"
        }
    ]
    
    return ideas

def generate_hypothesis_data():
    """Генерация данных для проверки гипотезы"""
    
    # Описательная статистика
    stats_summary = pd.DataFrame({
        'С курсами': group_with_courses[['math score', 'reading score', 'writing score', 'average_score']].mean(),
        'Без курсов': group_without_courses[['math score', 'reading score', 'writing score', 'average_score']].mean(),
        'Разница': group_with_courses[['math score', 'reading score', 'writing score', 'average_score']].mean() - 
                   group_without_courses[['math score', 'reading score', 'writing score', 'average_score']].mean(),
        'Прирост %': ((group_with_courses[['math score', 'reading score', 'writing score', 'average_score']].mean() - 
                       group_without_courses[['math score', 'reading score', 'writing score', 'average_score']].mean()) / 
                      group_without_courses[['math score', 'reading score', 'writing score', 'average_score']].mean() * 100)
    }).round(2)
    
    # T-тесты
    t_tests = {}
    for subject in ['math score', 'reading score', 'writing score', 'average_score']:
        t_stat, p_value = stats.ttest_ind(
            group_with_courses[subject].dropna(),
            group_without_courses[subject].dropna(),
            equal_var=False
        )
        t_tests[subject] = {
            't_stat': round(t_stat, 4),
            'p_value': round(p_value, 6),
            'significant': p_value < 0.05,
            'mean_diff': round(group_with_courses[subject].mean() - group_without_courses[subject].mean(), 2) if p_value < 0.05 else 0
        }
    
    # Дополнительные метрики
    target_with_courses = len(group_with_courses[group_with_courses['average_score'] >= 60]) / len(group_with_courses) * 100
    target_without_courses = len(group_without_courses[group_without_courses['average_score'] >= 60]) / len(group_without_courses) * 100
    
    # Анализ по уровню образования
    edu_level_analysis = cleaned_hypothesis_data.groupby('parental level of education').agg({
        'average_score': 'mean',
        'took_prep_course': 'mean',
        'total_score': 'count'
    }).round(2).reset_index()
    
    edu_level_analysis = edu_level_analysis.rename(columns={
        'average_score': 'Средний балл',
        'took_prep_course': 'Доля прошедших курсы',
        'total_score': 'Количество'
    })
    
    return {
        'stats_summary': stats_summary.to_dict(),
        't_tests': t_tests,
        'target_achievement': {
            'with_courses': round(target_with_courses, 1),
            'without_courses': round(target_without_courses, 1),
            'difference': round(target_with_courses - target_without_courses, 1)
        },
        'edu_level_analysis': edu_level_analysis.to_dict('records'),
        'group_sizes': {
            'with_courses': len(group_with_courses),
            'without_courses': len(group_without_courses),
            'total_non_higher_ed': len(cleaned_hypothesis_data)
        }
    }

def generate_visualizations():
    """Генерация всех визуализаций"""
    visualizations = {}
    
    # 1. Гистограмма распределения средних баллов
    fig1 = Figure(figsize=(10, 6))
    ax1 = fig1.subplots()
    ax1.hist([group_without_courses['average_score'], group_with_courses['average_score']], 
            bins=20, alpha=0.7, label=['Без курсов', 'С курсами'], color=['#FF9999', '#66B2FF'])
    ax1.set_title('Распределение средних баллов', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Средний балл')
    ax1.set_ylabel('Количество абитуриентов')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    visualizations['histogram'] = create_base64_plot(fig1)
    
    # 2. Box plot сравнения групп
    fig2 = Figure(figsize=(8, 6))
    ax2 = fig2.subplots()
    box_data = [group_without_courses['average_score'], group_with_courses['average_score']]
    ax2.boxplot(box_data, labels=['Без курсов', 'С курсами'], patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='darkblue'),
               medianprops=dict(color='red'))
    ax2.set_title('Сравнение средних баллов', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Средний балл')
    ax2.grid(True, alpha=0.3)
    visualizations['boxplot'] = create_base64_plot(fig2)
    
    # 3. Столбчатая диаграмма по предметам
    fig3 = Figure(figsize=(10, 6))
    ax3 = fig3.subplots()
    subjects = ['math score', 'reading score', 'writing score']
    x = np.arange(len(subjects))
    width = 0.35
    
    with_course_means = [group_with_courses[subject].mean() for subject in subjects]
    without_course_means = [group_without_courses[subject].mean() for subject in subjects]
    
    bars1 = ax3.bar(x - width/2, without_course_means, width, label='Без курсов', color='#FF9999')
    bars2 = ax3.bar(x + width/2, with_course_means, width, label='С курсами', color='#66B2FF')
    
    ax3.set_xlabel('Предметы')
    ax3.set_ylabel('Средний балл')
    ax3.set_title('Средние баллы по предметам', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Математика', 'Чтение', 'Письмо'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    visualizations['subject_comparison'] = create_base64_plot(fig3)
    
    # 4. Доля достигших целевого показателя
    fig4 = Figure(figsize=(8, 6))
    ax4 = fig4.subplots()
    
    categories = ['Достигли 60+', 'Не достигли 60+']
    with_course_counts = [
        len(group_with_courses[group_with_courses['average_score'] >= 60]),
        len(group_with_courses[group_with_courses['average_score'] < 60])
    ]
    without_course_counts = [
        len(group_without_courses[group_without_courses['average_score'] >= 60]),
        len(group_without_courses[group_without_courses['average_score'] < 60])
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, without_course_counts, width, label='Без курсов', color='#FF9999')
    bars2 = ax4.bar(x + width/2, with_course_counts, width, label='С курсами', color='#66B2FF')
    
    ax4.set_xlabel('Результат')
    ax4.set_ylabel('Количество абитуриентов')
    ax4.set_title('Достижение целевого показателя (60+)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    visualizations['target_achievement'] = create_base64_plot(fig4)
    
    # 5. Корреляционная матрица
    fig5 = Figure(figsize=(8, 6))
    ax5 = fig5.subplots()
    
    corr_matrix = cleaned_hypothesis_data[['math score', 'reading score', 'writing score', 
                                           'average_score', 'took_prep_course']].corr()
    
    im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    ax5.set_title('Корреляционная матрица', fontsize=14, fontweight='bold')
    ax5.set_xticks(range(len(corr_matrix.columns)))
    ax5.set_yticks(range(len(corr_matrix.columns)))
    ax5.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax5.set_yticklabels(corr_matrix.columns)
    
    # Добавляем значения в ячейки
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
    
    fig5.colorbar(im, ax=ax5)
    visualizations['correlation_matrix'] = create_base64_plot(fig5)
    
    return visualizations

@app.route('/')
def index():
    """Главная страница"""
    dashboard_data = generate_dashboard_data()
    return render_template('index.html', data=dashboard_data)

@app.route('/ideas')
def ideas():
    """Страница с идеями продуктов"""
    ideas_data = generate_ideas_data()
    return render_template('ideas.html', ideas=ideas_data)

@app.route('/hypothesis')
def hypothesis():
    """Страница с проверкой гипотезы"""
    hypothesis_data = generate_hypothesis_data()
    return render_template('hypothesis.html', data=hypothesis_data)

@app.route('/visualization')
def visualization():
    """Страница с визуализациями"""
    visualizations = generate_visualizations()
    return render_template('visualization.html', visualizations=visualizations)

@app.route('/recommendations')
def recommendations():
    """Страница с рекомендациями"""
    hypothesis_data = generate_hypothesis_data()
    stats_summary = pd.DataFrame(hypothesis_data['stats_summary'])
    
    # Расчет прироста
    max_diff_subject = ['математике', 'чтении', 'письме'][
        np.argmax([
            stats_summary.loc['math score', 'Разница'],
            stats_summary.loc['reading score', 'Разница'],
            stats_summary.loc['writing score', 'Разница']
        ])
    ]
    
    recommendations_data = {
        'avg_score_diff': round(stats_summary.loc['average_score', 'Разница'], 1),
        'target_diff': hypothesis_data['target_achievement']['difference'],
        'max_diff_subject': max_diff_subject,
        'potential_students': hypothesis_data['group_sizes']['without_courses'],
        'growth_percentage': round(stats_summary.loc['average_score', 'Прирост %'], 1)
    }
    
    return render_template('recommendations.html', data=recommendations_data)

@app.route('/api/dashboard')
def api_dashboard():
    """API для данных дашборда"""
    data = generate_dashboard_data()
    return jsonify(data)

@app.route('/api/ideas')
def api_ideas():
    """API для идей"""
    data = generate_ideas_data()
    return jsonify(data)

@app.route('/api/hypothesis')
def api_hypothesis():
    """API для гипотезы"""
    data = generate_hypothesis_data()
    return jsonify(data)

if __name__ == '__main__':
    # Загружаем данные при старте
    load_and_prepare_data()
    app.run(debug=True, port=5000)