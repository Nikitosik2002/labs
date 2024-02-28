import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка данных из файла 'diabetes.csv' с разделителем '\t' (табуляция)
data = pd.read_csv('diabetes.csv', delimiter='\t')

# Просмотр первых нескольких записей данных для ознакомления
print(data.head())

# Выделение матрицы признаков (X) и вектора меток (y)
X = data.drop(columns=['Диагноз']).values  # матрица признаков без столбца 'Диагноз'
y = data['Диагноз'].values  # вектор меток (классов)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Определение класса для логистической регрессии
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate  # коэффициент обучения
        self.n_iterations = n_iterations    # количество итераций
        self.weights = None  # веса
        self.bias = None     # смещение

    # Функция сигмоиды Принимает входное значение z.
    # Возвращает значение сигмоидной функции, которая преобразует входное значение в диапазон между 0 и 1.
    # Эта функция стремится минимизировать потери между предсказанными вероятностями
    # и истинными метками, что позволяет модели находить оптимальные параметры во время обучения.
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Функция потерь (логистическая функция потерь) Принимает предсказанные значения ℎ (вероятности) и истинные метки y
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # Обучение модели
    def fit(self, X, y):
        n_samples, n_features = X.shape  # количество образцов и признаков
        self.weights = np.zeros(n_features)  # инициализация весов нулями
        self.bias = 0  # инициализация смещения нулем

        for _ in range(self.n_iterations):
            # Вычисление значений z и y_pred
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # Вычисление градиентов
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Обновление весов и смещения
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Предсказание меток для новых данных
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred > 0.5).astype(int)

# Создание и обучение модели логистической регрессии на исходных данных
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказание меток для тестовых данных
y_pred = model.predict(X_test)

# Оценка точности классификации на тестовых данных
accuracy = accuracy_score(y_test, y_pred)
print("Точность до отбора признаков:", accuracy)

# Вычисление матрицы корреляции между признаками
correlation_matrix = data.corr().abs()

# Выбор верхнего треугольника матрицы корреляции (без диагонали)
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Нахождение признаков с высокой корреляцией
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

# Удаление признаков с высокой корреляцией из данных
data_filtered = data.drop(columns=high_corr_features)

# Выделение новой матрицы признаков и вектора меток
X_filtered = data_filtered.drop(columns=['Диагноз']).values

# Разделение отфильтрованных данных на обучающую и тестовую выборки
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y, test_size=0.4, random_state=100)

# Создание и обучение новой модели на отфильтрованных данных
model_filtered = LogisticRegression()
model_filtered.fit(X_train_filtered, y_train_filtered)

# Предсказание меток для тестовых данных после отбора признаков
y_pred_filtered = model_filtered.predict(X_test_filtered)

# Оценка точности классификации после отбора признаков
accuracy_filtered = accuracy_score(y_test_filtered, y_pred_filtered)
print("Точность после отбора признаков:", accuracy_filtered)
