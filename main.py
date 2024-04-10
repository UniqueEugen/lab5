import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Загрузка данных из CSV-файла
data = pd.read_csv('Mall_Customers.csv')

# Разделение данных на признаки и целевую переменную
X = data[["Age", "Annual Income (k$)"]]
y = data["Genre"]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели наивного байесовского классификатора
model = GaussianNB()
model.fit(X_train, y_train)

# Предсказание пола для тестового набора данных
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Определение цветов для каждого класса
colors = {"Male": "blue", "Female": "red"}

# Визуализация результатов
plt.figure(figsize=(8, 6))

# Отображение предсказанных значений (y_pred)
for genre in colors:
    mask_pred = (y_pred == genre)
    plt.scatter(X_test.loc[mask_pred, "Age"], X_test.loc[mask_pred, "Annual Income (k$)"], color=colors[genre], marker='o', label=f"Predicted {genre}")

# Отображение фактических значений (y_test)
for genre in colors:
    mask_true = (y_test == genre)
    plt.scatter(X_test.loc[mask_true, "Age"], X_test.loc[mask_true, "Annual Income (k$)"], color=colors[genre], marker='x', label=f"Actual {genre}")

plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.legend()
plt.show()