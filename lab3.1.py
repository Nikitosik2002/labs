import numpy as np  # Импорт библиотеки NumPy для работы с массивами данных
# Импорт функции для разделения данных на обучающий и тестовый наборы
from sklearn.model_selection import train_test_split
# Импорт функции для вычисления точности модели
from sklearn.metrics import accuracy_score

# Загрузка данных из файла "diabetes.csv", где каждая строка представляет собой набор признаков и целевую переменную
data = np.loadtxt('diabetes.csv', delimiter='\t', skiprows=1)

# Разделение данных на матрицу признаков (X) и вектор целевой переменной (y)
X = data[:, :-1]  # Признаки - все столбцы, кроме последнего
y = data[:, -1]  # Целевая переменная - последний столбец

# Вычисление матрицы корреляций между признаками и целевой переменной
correlation_matrix = np.corrcoef(X.T, y)

# Находим индексы признаков с наибольшей корреляцией с целевой переменной
selected_feature_indices = np.argsort(
    np.abs(correlation_matrix[-1, :-1]))[::-1][:2]

# Выбираем только признаки с наибольшей корреляцией
X_selected = X[:, selected_feature_indices]

# Добавляем столбец единиц для учета свободного коэффициента
X_selected = np.hstack((np.ones((X_selected.shape[0], 1)), X_selected))

# Разделение выборки на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42)

# Определяем функцию логистической активации (сигмоида)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Метод градиентного спуска для обучения модели логистической регрессии


def gradient_descent(X, y, alpha, epochs):
    m, n = X.shape  # Количество примеров и признаков
    theta = np.zeros(n)  # Инициализация параметров модели нулями

    # Итерационный процесс обновления параметров модели
    for _ in range(epochs):
        # Вычисление линейной комбинации признаков и параметров модели
        z = np.dot(X, theta)
        h = sigmoid(z)  # Применение сигмоидной функции активации
        # Вычисление градиента функции потерь
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient  # Обновление параметров модели
    return theta


# Обучение модели на обучающем наборе данных
alpha = 0.01  # Скорость обучения
epochs = 1000  # Количество итераций обновления параметров
theta = gradient_descent(X_train, y_train, alpha, epochs)  # Обучение модели

# Прогнозирование на обучающем наборе данных
# Применение модели к обучающим данным
train_predictions = sigmoid(np.dot(X_train, theta))
# Преобразование вероятностей в классы
train_predictions = np.where(train_predictions >= 0.5, 1, 0)

# Прогнозирование на тестовом наборе данных
# Применение модели к тестовым данным
test_predictions = sigmoid(np.dot(X_test, theta))
# Преобразование вероятностей в классы
test_predictions = np.where(test_predictions >= 0.5, 1, 0)

# Оценка точности классификации на обучающем наборе данных
# Вычисление точности модели на обучающих данных
train_accuracy = accuracy_score(y_train, train_predictions)
print("Точность классификации на обучающем наборе данных:", train_accuracy)

# Оценка точности классификации на тестовом наборе данных
# Вычисление точности модели на тестовых данных
test_accuracy = accuracy_score(y_test, test_predictions)
print("Точность классификации на тестовом наборе данных:", test_accuracy)
