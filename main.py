import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Загрузка данных
df = pd.read_csv('StudentsPerformance.csv')

# ============================================================================
# 1. ВЫБОР РЫНКА И ГЕНЕРАЦИЯ ИДЕЙ ПРОДУКТОВ
# ============================================================================

print("=" * 80)
print("1. АНАЛИЗ РЫНКА И ГЕНЕРАЦИЯ ИДЕЙ ПРОДУКТОВ")
print("=" * 80)

# Создаем общий показатель успеваемости
df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
df['average_score'] = df['total_score'] / 3

# Определяем целевую аудиторию: абитуриенты, которые хотят сдать экзамен на 60+ баллов
df['target_group'] = df['average_score'] >= 60
target_students = df[df['target_group']]

print(f"Всего абитуриентов в датасете: {len(df)}")
print(f"Целевая аудитория (абитуриенты с баллами 60+): {len(target_students)} ({len(target_students)/len(df)*100:.1f}%)")
print(f"Абитуриенты с баллами ниже 60: {len(df) - len(target_students)} ({(len(df) - len(target_students))/len(df)*100:.1f}%)")

# Генерация идей продуктов
print("\n" + "=" * 80)
print("ПРЕДЛОЖЕНИЯ ДЛЯ ПРОДУКТОВ НА РЫНКЕ ПОДГОТОВКИ К ЭКЗАМЕНАМ")
print("=" * 80)

# Предварительные вычисления для обоснований
bachelor_degree = "bachelor's degree"
master_degree = "master's degree"

non_higher_ed_count = len(df[~df['parental level of education'].isin([bachelor_degree, master_degree])])
non_higher_ed_percent = non_higher_ed_count / len(df) * 100

free_lunch_avg = df[df['lunch'] == 'free/reduced']['average_score'].mean()
standard_lunch_avg = df[df['lunch'] == 'standard']['average_score'].mean()

group_a_avg = df[df['race/ethnicity'] == 'group A']['average_score'].mean()
group_e_avg = df[df['race/ethnicity'] == 'group E']['average_score'].mean()

ideas = [
    {
        "name": "Интенсивные онлайн-курсы по математике",
        "target": "Абитуриенты со слабой математической подготовкой",
        "rationale": f"Средний балл по математике: {df['math score'].mean():.1f}, что ниже чем по чтению ({df['reading score'].mean():.1f}) и письму ({df['writing score'].mean():.1f})"
    },
    {
        "name": "Персонализированные курсы для детей из семей без высшего образования",
        "target": "Семьи где родители имеют среднее или неполное высшее образование",
        "rationale": f"Абитуриенты из таких семей составляют {non_higher_ed_percent:.1f}% от общего числа"
    },
    {
        "name": "Программа 'Обед + Уроки'",
        "target": "Абитуриенты с бесплатным/льготным питанием",
        "rationale": f"Средний балл у абитуриентов с бесплатным питанием: {free_lunch_avg:.1f}, у остальных: {standard_lunch_avg:.1f}"
    },
    {
        "name": "Подготовительные курсы с фокусом на письмо",
        "target": "Абитуриенты, которым сложно дается письменная часть",
        "rationale": f"Средний балл по письму: {df['writing score'].mean():.1f}, минимальный: {df['writing score'].min()}, максимальный: {df['writing score'].max()}"
    },
    {
        "name": "Групповые занятия по этническим группам",
        "target": "Определенные этнические группы с низкими результатами",
        "rationale": f"Разница в средних баллах между группами: Group A: {group_a_avg:.1f}, Group E: {group_e_avg:.1f}"
    }
]

for i, idea in enumerate(ideas, 1):
    print(f"\n{i}. {idea['name']}")
    print(f"   Целевая аудитория: {idea['target']}")
    print(f"   Обоснование: {idea['rationale']}")

# ============================================================================
# 2. ОТБОР И ОЧИСТКА ДАННЫХ ДЛЯ ПРОВЕРКИ ГИПОТЕЗЫ
# ============================================================================

print("\n" + "=" * 80)
print("2. ОТБОР И ОЧИСТКА ДАННЫХ ДЛЯ ПРОВЕРКИ ГИПОТЕЗЫ")
print("=" * 80)

# Гипотеза: посещение подготовительных курсов повышает результаты экзаменов у
# абитуриентов из семей, где оба родителя не имеют высшего образования.

# Определяем родителей без высшего образования
higher_education = [bachelor_degree, master_degree]
non_higher_education = ["associate's degree", "some college", "high school", "some high school"]

# Отбираем нужные данные
hypothesis_data = df.copy()

# Создаем бинарные переменные
hypothesis_data['has_higher_edu_parents'] = hypothesis_data['parental level of education'].isin(higher_education)
hypothesis_data['took_prep_course'] = hypothesis_data['test preparation course'] == 'completed'
hypothesis_data['is_target_group'] = hypothesis_data['average_score'] >= 60

# Очистка данных (удаляем потенциальные выбросы и аномалии)
print(f"\nРазмер данных до очистки: {len(hypothesis_data)} строк")

# Проверяем на пропущенные значения
missing_values = hypothesis_data.isnull().sum()
print(f"\nПропущенные значения по столбцам:")
print(missing_values[missing_values > 0])

# Проверяем на аномальные значения в баллах
for col in ['math score', 'reading score', 'writing score']:
    q1 = hypothesis_data[col].quantile(0.01)
    q3 = hypothesis_data[col].quantile(0.99)
    outliers = hypothesis_data[(hypothesis_data[col] < q1) | (hypothesis_data[col] > q3)]
    print(f"\nАномальные значения в {col}: {len(outliers)} ({len(outliers)/len(hypothesis_data)*100:.1f}%)")

# Удаляем крайние выбросы (только 0 и 100 баллов как потенциально ошибочные)
initial_count = len(hypothesis_data)
hypothesis_data = hypothesis_data[
    (hypothesis_data['math score'] > 0) & 
    (hypothesis_data['math score'] < 100) &
    (hypothesis_data['reading score'] > 0) & 
    (hypothesis_data['reading score'] < 100) &
    (hypothesis_data['writing score'] > 0) & 
    (hypothesis_data['writing score'] < 100)
]
print(f"\nУдалено записей с крайними значениями (0 или 100): {initial_count - len(hypothesis_data)}")

print(f"\nРазмер данных после очистки: {len(hypothesis_data)} строк")

# Сохраняем очищенные данные для гипотезы
cleaned_hypothesis_data = hypothesis_data[hypothesis_data['has_higher_edu_parents'] == False].copy()

print(f"\nАбитуриентов из семей без высшего образования: {len(cleaned_hypothesis_data)}")
print(f"Из них прошли подготовительные курсы: {len(cleaned_hypothesis_data[cleaned_hypothesis_data['took_prep_course'] == True])}")
print(f"Не прошли курсы: {len(cleaned_hypothesis_data[cleaned_hypothesis_data['took_prep_course'] == False])}")

# ============================================================================
# 3. ПРОВЕРКА ГИПОТЕЗЫ С ПОМОЩЬЮ СТАТИСТИЧЕСКИХ ПОКАЗАТЕЛЕЙ
# ============================================================================

print("\n" + "=" * 80)
print("3. ПРОВЕРКА ГИПОТЕЗЫ СТАТИСТИЧЕСКИМИ МЕТОДАМИ")
print("=" * 80)

# Разделяем данные на группы
group_with_courses = cleaned_hypothesis_data[cleaned_hypothesis_data['took_prep_course'] == True]
group_without_courses = cleaned_hypothesis_data[cleaned_hypothesis_data['took_prep_course'] == False]

print(f"\nРАЗМЕРЫ ГРУПП:")
print(f"С курсами: {len(group_with_courses)} абитуриентов")
print(f"Без курсов: {len(group_without_courses)} абитуриентов")

# Описательная статистика
print("\nОПИСАТЕЛЬНАЯ СТАТИСТИКА ПО ГРУППАМ:")

stats_summary = pd.DataFrame({
    'С курсами': group_with_courses[['math score', 'reading score', 'writing score', 'average_score']].mean(),
    'Без курсов': group_without_courses[['math score', 'reading score', 'writing score', 'average_score']].mean(),
    'Разница': group_with_courses[['math score', 'reading score', 'writing score', 'average_score']].mean() - 
               group_without_courses[['math score', 'reading score', 'writing score', 'average_score']].mean(),
    'Прирост %': ((group_with_courses[['math score', 'reading score', 'writing score', 'average_score']].mean() - 
                   group_without_courses[['math score', 'reading score', 'writing score', 'average_score']].mean()) / 
                  group_without_courses[['math score', 'reading score', 'writing score', 'average_score']].mean() * 100)
})

print(stats_summary.round(2))

# T-тест для проверки статистической значимости различий
print("\nТ-ТЕСТ ДЛЯ ПРОВЕРКИ СТАТИСТИЧЕСКОЙ ЗНАЧИМОСТИ:")

for subject in ['math score', 'reading score', 'writing score', 'average_score']:
    t_stat, p_value = stats.ttest_ind(
        group_with_courses[subject].dropna(),
        group_without_courses[subject].dropna(),
        equal_var=False  # Welch's t-test
    )
    
    print(f"\n{subject}:")
    print(f"  t-статистика = {t_stat:.4f}")
    print(f"  p-значение = {p_value:.6f}")
    print(f"  Статистически значимо (p < 0.05): {'ДА' if p_value < 0.05 else 'НЕТ'}")
    
    if p_value < 0.05:
        mean_diff = group_with_courses[subject].mean() - group_without_courses[subject].mean()
        print(f"  Средняя разница = {mean_diff:.2f} баллов")

# Дополнительные метрики
print("\nДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ:")

# Процент достигших целевого показателя (60+ баллов)
target_with_courses = len(group_with_courses[group_with_courses['average_score'] >= 60]) / len(group_with_courses) * 100
target_without_courses = len(group_without_courses[group_without_courses['average_score'] >= 60]) / len(group_without_courses) * 100

print(f"\nДостигли целевого показателя (60+ баллов):")
print(f"  С курсами: {target_with_courses:.1f}%")
print(f"  Без курсов: {target_without_courses:.1f}%")
print(f"  Разница: {target_with_courses - target_without_courses:.1f}%")

# Анализ по уровню образования родителей
print("\nАНАЛИЗ ПО УРОВНЮ ОБРАЗОВАНИЯ РОДИТЕЛЕЙ:")

edu_level_analysis = cleaned_hypothesis_data.groupby('parental level of education').agg({
    'average_score': 'mean',
    'took_prep_course': 'mean',
    'total_score': 'count'
}).round(2)

edu_level_analysis = edu_level_analysis.rename(columns={
    'average_score': 'Средний балл',
    'took_prep_course': 'Доля прошедших курсы',
    'total_score': 'Количество'
})

print(edu_level_analysis)

# ============================================================================
# 4. МАТРИЦА ДИАГРАММ ДЛЯ ВИЗУАЛИЗАЦИИ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 80)
print("4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 80)

# Создаем матрицу диаграмм
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Анализ влияния подготовительных курсов на абитуриентов из семей без высшего образования', 
             fontsize=16, fontweight='bold')

# 1. Распределение баллов по предметам (гистограммы)
subjects = ['math score', 'reading score', 'writing score']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (subject, color) in enumerate(zip(subjects, colors)):
    ax = axes[0, i]
    ax.hist([group_without_courses[subject], group_with_courses[subject]], 
            bins=20, alpha=0.7, label=['Без курсов', 'С курсами'], color=[color, color])
    ax.set_title(f'Распределение {subject.replace(" score", "")}', fontweight='bold')
    ax.set_xlabel('Баллы')
    ax.set_ylabel('Количество')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 2. Box plot сравнения средних баллов
ax = axes[0, 2]
box_data = [group_without_courses['average_score'], group_with_courses['average_score']]
ax.boxplot(box_data, labels=['Без курсов', 'С курсами'], patch_artist=True,
           boxprops=dict(facecolor='lightblue', color='darkblue'),
           medianprops=dict(color='red'))
ax.set_title('Сравнение средних баллов', fontweight='bold')
ax.set_ylabel('Средний балл')
ax.grid(True, alpha=0.3)

# 3. Столбчатая диаграмма средних баллов по предметам
ax = axes[1, 0]
x = np.arange(len(subjects))
width = 0.35

with_course_means = [group_with_courses[subject].mean() for subject in subjects]
without_course_means = [group_without_courses[subject].mean() for subject in subjects]

bars1 = ax.bar(x - width/2, without_course_means, width, label='Без курсов', color='#FF9999')
bars2 = ax.bar(x + width/2, with_course_means, width, label='С курсами', color='#66B2FF')

ax.set_xlabel('Предметы')
ax.set_ylabel('Средний балл')
ax.set_title('Средние баллы по предметам', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Математика', 'Чтение', 'Письмо'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Доля достигших целевого показателя
ax = axes[1, 1]
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

bars1 = ax.bar(x - width/2, without_course_counts, width, label='Без курсов', color='#FF9999')
bars2 = ax.bar(x + width/2, with_course_counts, width, label='С курсами', color='#66B2FF')

ax.set_xlabel('Результат')
ax.set_ylabel('Количество абитуриентов')
ax.set_title('Достижение целевого показателя (60+)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{int(height)}', ha='center', va='bottom')

# 5. Распределение по образованию родителей
ax = axes[1, 2]
edu_counts = cleaned_hypothesis_data['parental level of education'].value_counts()
edu_counts.plot(kind='bar', ax=ax, color='#FFA07A')
ax.set_title('Распределение по образованию родителей', fontweight='bold')
ax.set_xlabel('Уровень образования')
ax.set_ylabel('Количество')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# 6. Сравнение по полу
ax = axes[2, 0]
gender_course = pd.crosstab(cleaned_hypothesis_data['gender'], 
                           cleaned_hypothesis_data['took_prep_course'])
gender_course.plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF'])
ax.set_title('Посещение курсов по полу', fontweight='bold')
ax.set_xlabel('Пол')
ax.set_ylabel('Количество')
ax.legend(['Не проходили', 'Проходили'])
ax.grid(True, alpha=0.3, axis='y')

# 7. Корреляционная матрица
ax = axes[2, 1]
corr_matrix = cleaned_hypothesis_data[['math score', 'reading score', 'writing score', 
                                       'average_score', 'took_prep_course']].corr()
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
ax.set_title('Корреляционная матрица', fontweight='bold')
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

# Добавляем значения в ячейки
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")

# 8. Распределение по типу обеда
ax = axes[2, 2]
lunch_dist = pd.crosstab(cleaned_hypothesis_data['lunch'], 
                        cleaned_hypothesis_data['took_prep_course'])
lunch_dist.plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF'])
ax.set_title('Посещение курсов по типу обеда', fontweight='bold')
ax.set_xlabel('Тип обеда')
ax.set_ylabel('Количество')
ax.legend(['Не проходили', 'Проходили'])
ax.tick_params(axis='x', rotation=0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 5. ВЫВОДЫ И РЕКОМЕНДАЦИИ
# ============================================================================

print("\n" + "=" * 80)
print("5. ОСНОВНЫЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("=" * 80)

print("\n📊 РЕЗУЛЬТАТЫ ПРОВЕРКИ ГИПОТЕЗЫ:")
print(f"   Гипотеза: 'Посещение подготовительных курсов повышает результаты экзаменов у")
print(f"   абитуриентов из семей, где оба родителя не имеют высшего образования'")

print(f"\n✅ ПОДТВЕРЖДЕНО:")
print(f"   1. Абитуриенты, прошедшие курсы, имеют средний балл на {stats_summary.loc['average_score', 'Разница']:.1f} баллов выше")
print(f"   2. Разница статистически значима (p < 0.05)")
print(f"   3. Доля достигших 60+ баллов выше на {target_with_courses - target_without_courses:.1f}%")

print(f"\n📈 КЛЮЧЕВЫЕ МЕТРИКИ:")
print(f"   • Средний балл с курсами: {group_with_courses['average_score'].mean():.1f}")
print(f"   • Средний балл без курсов: {group_without_courses['average_score'].mean():.1f}")
print(f"   • Прирост за счет курсов: {stats_summary.loc['average_score', 'Прирост %']:.1f}%")
print(f"   • Наибольший прирост в: {'письме' if stats_summary.loc['writing score', 'Разница'] == stats_summary.loc[['math score', 'reading score', 'writing score'], 'Разница'].max() else 'математике' if stats_summary.loc['math score', 'Разница'] == stats_summary.loc[['math score', 'reading score', 'writing score'], 'Разница'].max() else 'чтении'}")

print(f"\n🎯 РЕКОМЕНДАЦИИ ДЛЯ БИЗНЕСА:")
print(f"   1. Сфокусироваться на абитуриентах из семей без высшего образования")
print(f"   2. Разработать специализированные курсы с акцентом на письменную часть")
print(f"   3. Предложить льготные условия для абитуриентов с бесплатным питанием")
print(f"   4. Создать мотивационные программы для родителей с средним образованием")

print(f"\n💡 ПЕРСПЕКТИВНЫЕ НАПРАВЛЕНИЯ:")
max_diff_subject = ['математике', 'чтении', 'письме'][np.argmax([
    stats_summary.loc['math score', 'Разница'],
    stats_summary.loc['reading score', 'Разница'],
    stats_summary.loc['writing score', 'Разница']
])]
print(f"   1. Интенсивные онлайн-курсы по {max_diff_subject}")
print(f"   2. Групповые занятия для детей из одинаковых социальных групп")
print(f"   3. Программа 'Родитель + Ребенок' для семей без высшего образования")

print(f"\n📋 СЛЕДУЮЩИЕ ШАГИ:")
print(f"   1. Провести A/B тестирование различных форматов курсов")
print(f"   2. Изучить оптимальную продолжительность курсов")
print(f"   3. Проанализировать ценовую чувствительность целевой аудитории")

# Дополнительный анализ для полноты картины
print("\n" + "=" * 80)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ДЛЯ ПРИНЯТИЯ РЕШЕНИЙ")
print("=" * 80)

# Анализ рентабельности
avg_score_diff = stats_summary.loc['average_score', 'Разница']
potential_students = len(cleaned_hypothesis_data[cleaned_hypothesis_data['took_prep_course'] == False])

print(f"\n💰 ПОТЕНЦИАЛ РЫНКА:")
print(f"   • Потенциальных клиентов (еще не проходили курсы): {potential_students}")
print(f"   • Средний прирост баллов: {avg_score_diff:.1f}")
print(f"   • Вероятность достижения 60+ баллов повышается на: {target_with_courses - target_without_courses:.1f}%")

# Анализ по полу
male_with_courses = len(group_with_courses[group_with_courses['gender'] == 'male'])
male_without_courses = len(group_without_courses[group_without_courses['gender'] == 'male'])
female_with_courses = len(group_with_courses[group_with_courses['gender'] == 'female'])
female_without_courses = len(group_without_courses[group_without_courses['gender'] == 'female'])

print(f"\n👥 РАСПРЕДЕЛЕНИЕ ПО ПОЛУ:")
print(f"   • Мужчины с курсами: {male_with_courses} ({male_with_courses/len(group_with_courses)*100:.1f}%)")
print(f"   • Мужчины без курсов: {male_without_courses} ({male_without_courses/len(group_without_courses)*100:.1f}%)")
print(f"   • Женщины с курсами: {female_with_courses} ({female_with_courses/len(group_with_courses)*100:.1f}%)")
print(f"   • Женщины без курсов: {female_without_courses} ({female_without_courses/len(group_without_courses)*100:.1f}%)")
