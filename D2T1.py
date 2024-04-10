import numpy as np
import pandas as pd
import sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sns

# Загрузка данных из CSV-файла
data = pd.read_csv('winequality-white.csv', delimiter=';')

# Разделение данных на признаки и целевую переменную
X = data.drop('quality', axis=1)
y = np.where(data['quality'] >= 7, 'хороший', 'плохой')

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели наивного байесовского классификатора
model = GaussianNB()
model.fit(X_train, y_train)

# Предсказание типа вина в тестовом наборе данных
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Визуализация
fig, ax = plt.subplots(figsize=(8, 6))

# Распределение реальных значений
sns.histplot(data=y_test, kde=True, color='blue', label='Реальные значения')

# Распределение предсказанных значений
sns.histplot(data=y_pred, kde=True, color='orange', label='Предсказанные значения')

ax.set_title('Распределение реальных и предсказанных значений')
ax.set_xlabel('Quality')
ax.set_ylabel('Count')
ax.legend()

plt.show()