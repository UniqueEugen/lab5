import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Загрузка данных из CSV-файла
data = pd.read_csv('Mall_Customers.csv')

# Разделение данных на признаки (возраст и годовой доход) и целевую переменную (уровень оценки трат)
X = data[['Age', 'Annual Income (k$)']]
y = data['Spending Score (1-100)']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели наивного байесовского классификатора
model = GaussianNB()
model.fit(X_train, y_train)

# Предсказание уровней оценки трат покупателей в тестовом наборе данных
y_pred = model.predict(X_test)

# Визуализация результатов классификации
scatter = plt.scatter(X_test['Age'], X_test['Annual Income (k$)'], c=y_pred, cmap='RdYlBu')

# Создание градиентной легенды
cmap = LinearSegmentedColormap.from_list("", ["darkviolet", "yellow"])
cbar = plt.colorbar(scatter)
cbar.set_ticks([y_pred.min(), y_pred.max()])
cbar.set_ticklabels(['1', '100'])
cbar.set_label('Spending Score', rotation=270, labelpad=15)

plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Classification Results - Spending Score')
plt.show()